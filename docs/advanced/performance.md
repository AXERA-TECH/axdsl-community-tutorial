# 性能优化指南

性能优化应建立在正确性之上。推荐循环：

```text
确认正确性
  ↓
记录 baseline
  ↓
导出性能报告
  ↓
定位瓶颈
  ↓
只改一个变量
  ↓
复测 correctness + performance
```

不要在没有 trace / cycle / traffic 证据的情况下盲目调 tile、展开循环或增加 core 数。

## 1. 建立性能基线

最小基线脚本应记录：

- 工具链版本 / commit
- input shape
- dtype
- grid
- core num
- tile size
- run mode
- cache 是否启用
- 输出目录
- simulator cycles
- correctness 结果

建议命名：

```text
cases/
  shape_1024x1024_bf16_tile_128.sh
  shape_4093x512_bf16_tile_64.sh
```

不要只用一个 shape 做结论。至少覆盖：

- 小 shape
- 典型 shape
- 大 shape
- 非整除 shape
- 尾块比例较大的 shape

## 2. 关键指标

| 指标 | 含义 | 常见问题 |
|---|---|---|
| estimated cycles | 估算执行周期 | 整体耗时 |
| job 数量 | 后端调度任务数 | 任务过碎、调度开销 |
| engine utilization | 计算引擎利用率 | 计算不足 |
| EU duty | 执行单元占用 | 空闲 / pipeline bubble |
| DDR traffic | DDR 读写量 | 重复 load / store |
| OCM usage | 片上内存占用 | tile 或临时变量过大 |
| memory bandwidth | 带宽压力 | 访存受限 |
| overlap | 搬运与计算重叠 | 串行等待 |
| idle / bubble | 空闲周期 | 同步或任务不均 |
| multi-core idle | core 空闲 | 负载不均 |

## 3. 性能工具

### 3.1 `dump_dsl_perf`

```python
@ax.builder(
    run_mode=ax.RunMode.CRUN,
    enable_cache=False,
    dump_dsl_perf=True,
    dump_path="output/my_kernel",
)
def kernel(...):
    ...
```

要求：

- CRUN
- cache 关闭
- 当前源码要求 single chip

可导出：

- DSL source snapshot
- raw / fused graph
- compiled job metadata
- simulator cycle trace

运行后日志会输出类似命令：

```bash
python3 .../perf_analyzer.py --artifact-dir output/my_kernel
```

用于生成 JSON / HTML 报告。

### 3.2 `DUMP_NPU_TRACE=1`

```bash
DUMP_NPU_TRACE=1 python3 my_kernel.py
```

产物：

```text
output/trace.json
```

适合查看：

- scheduled jobs
- job 依赖
- 计算与搬运 overlap
- engine 利用率
- pipeline bubble

### 3.3 `DEBUG_SIM_CYCLE_TRACE=1`

```bash
DEBUG_SIM_CYCLE_TRACE=1 python3 my_kernel.py
```

产物：

```text
output/simulator_cycle_trace.json
```

适合分析：

- 运行期控制流
- while / range 实际行为
- signal 等待
- 动态事件时序

### 3.4 build context

默认可导出：

```text
output/build_context.json
```

适合快速查看：

- jobs
- traffic
- memory usage
- estimated cycles
- EU duty

## 4. 性能瓶颈分类

### Compute-bound

特征：

- engine utilization / EU duty 高
- traffic 相对不高
- 增加 tile 对带宽帮助有限

优化：

- 使用后端高效算子
- 改善 dtype
- 融合 elementwise
- 提高 GEMM / Conv tile 效率
- 减少低效算子组合

### Memory-bound

特征：

- DDR traffic 高
- engine 空闲
- 搬运时间长

优化：

- 减少 load / store
- 提高 tile 复用
- 中间结果留在 OCM
- 改变 layout
- 合并小搬运

### Synchronization-bound

特征：

- signal wait 时间长
- core idle
- 多芯片 step 相互等待

优化：

- 减少同步次数
- 合并数据块
- 通信与计算 overlap
- 均衡任务
- 检查死锁或不必要 barrier

### Launch / scheduling overhead

特征：

- job 过碎
- 小 shape 上调度占比高

优化：

- 增大 tile
- 减少循环展开碎片
- 合并 elementwise
- 对 shape 做专用策略

## 5. Tile size 优化

Tile size 通常影响：

- OCM 使用
- DDR traffic
- 数据复用
- parallelism
- 尾块数量
- 后端 job 形态

### Elementwise

对 elementwise 而言，更大 tile 主要减少调度次数和搬运次数，但并行度可能下降。

### GEMM

GEMM 中 tile 复用更明显：

```text
A(M, K) @ B(K, N)
```

若每个 program 计算 `C(TM, TN)`：

- A tile 可在 N 方向复用
- B tile 可在 M 方向复用
- 更大 TM / TN 可能减少总 load
- 但 A + B + C + 临时区都必须放进 OCM

### 经验流程

```text
1. 从保守 tile 开始
2. 确保 correctness
3. 记录 cycles / traffic / OCM
4. 调整一个维度
5. 找到 OCM 允许的候选范围
6. 用真实 shape 复测
```

不要假设某个 tile size 对所有 shape 都最优。

## 6. 数据搬运优化

### 6.1 避免重复 load

如果多个算子使用同一个输入：

```python
x_tile = ax.dsl.load(x[start:stop])
a = ax.dsl.silu(x_tile)
b = ax.dsl.mul(a, x_tile)
```

不要重复 load 同一切片。

### 6.2 中间结果不要落 DDR

不必要模式：

```text
store temp0 → DDR
load temp0 → OCM
compute
store result
```

除非需要跨 program / core 交换，否则应保留在图中让优化器调度。

### 6.3 合并小操作

多个 elementwise 和 cast 可以融合：

```python
y = ax.dsl.cast(ax.dsl.mul(ax.dsl.add(x, one), scale), dtype=ax.bf16)
```

### 6.4 检查 layout

Transpose、reshape、broadcast 可能引发数据重排。结合 trace 判断它们是否带来额外 move。

## 7. 算子融合

常见融合机会：

- add + activation
- mul + add → FMA
- activation + elementwise
- reduce + elementwise
- normalize 相关模式
- attention 内部的 score / softmax / matmul
- matmul + activation

源码中已有图变换示例包括：

- `AxMul + AxAdd → FMA`
- `AxMul + AxReduceSum`
- `AxExp + AxReduceSum`
- `AxReduceMax + AxSub`

融合收益：

- 减少中间 tensor
- 减少 traffic
- 减少 job 数
- 提高数据局部性

注意：

- 融合可能改变浮点结合顺序
- 低精度中间结果可能影响精度
- per-layer check 可定位融合前后差异

## 8. 多核优化

### 提高利用率

- 每个 core 处理相近数据量
- 避免所有 core 等一个 core
- merge 操作可分两级
- 长尾任务单独拆分

### 减少同步

- 只在数据依赖处使用 signal
- 避免 `sync_all` 泛滥
- 用阶段 signal 替代多次全局 barrier
- 通信块不要过小

### 检查输出模式

如果输出写入导致冲突，会引入串行化或错误。使用明确 owner 或支持并发语义的 API。

## 9. 多芯片优化

### 通信与计算 overlap

避免：

```text
完整 all-gather
  ↓
完整 matmul
```

尽量：

```text
收到 block0 → matmul block0
同时发送下一 block
```

### 控制 chunk

| chunk 太小 | chunk 太大 |
|---|---|
| signal / job 开销高 | overlap 差 |
| 调度碎片多 | OCM 压力大 |
| 带宽利用率低 | 等待时间长 |

### 减少全局广播

只发送接收方需要的数据，避免为了代码简单广播完整 tensor。

## 10. 控制流优化

### 优先使用 `range`

如果循环次数可表示为 start / stop / step，优先 `range`，而不是 `while_loop`。`range` 更利于展开和静态调度。

### 谨慎使用 unroll_factor

增大 `unroll_factor` 可能提升调度机会，但也会：

- 增加 OCM
- 增大编译时间
- 恶化尾块
- 增大代码 / job 数量

应通过 trace 比较不同值。

### 减少循环内状态数量

loop-carried state 越多，依赖和调度越复杂。只保留真正需要跨迭代的状态。

## 11. 编译缓存

启用：

```python
@ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=True)
def kernel(...):
    ...
```

缓存 key 包含：

- kernel source / AST
- 参数 shape 和 dtype
- compile-time 参数值
- launch grid
- 工具链版本 / commit
- 编译配置
- 依赖版本

命中缓存时可以跳过 graph 生成和编译，直接恢复产物并仿真，显著减少启动时间。

限制：

- correctness check 与 cache 不能同时开启
- 调试 graph 时应关闭 cache，确保产物来自当前代码
- 工具链或依赖变化应使 cache key 变化

## 12. 自动调参状态说明

部分旧版 AXDSL 文档中提到 `auto_tune_cases`。在当前开源源码的公开 `ax.builder` 签名中，该参数已移除。

因此：

- 社区首版文档不应承诺该 API 可用
- 可在脚本层枚举 tile size / grid，并结合性能报告选择策略
- 后续工具链恢复 auto-tune 后，再补充官方章节

脚本层调参示例：

```python
CASES = [
    {"tile_m": 32, "tile_n": 64, "grid": ...},
    {"tile_m": 64, "tile_n": 64, "grid": ...},
    {"tile_m": 128, "tile_n": 64, "grid": ...},
]

for case in CASES:
    run_benchmark(case)
```

## 13. dtype 与性能

常见方向：

- fp32 用于语义验证
- bf16 / fp16 提高吞吐
- int8 / fp8 需要确认 scale 与后端支持

必须同时评估：

- speed
- accuracy
- numerical stability
- 后端支持矩阵
- reference 差异

不要为了性能直接降低 reduce / softmax 中间精度。

## 14. 性能实验模板

```python
CASES = [
    {"tile_m": 32, "tile_n": 64},
    {"tile_m": 64, "tile_n": 64},
    {"tile_m": 64, "tile_n": 128},
]

for case in CASES:
    tile_m = case["tile_m"]
    tile_n = case["tile_n"]
    grid = (ceil_div(M, tile_m), ceil_div(N, tile_n))

    # 1. 正确性
    # 2. dump build context / trace
    # 3. 记录 cycles, traffic, OCM
    # 4. 保存输出目录
```

报告表格建议：

| Case | Shape | dtype | Grid | Tile | Cycles | DDR traffic | OCM | Correct |
|---|---|---|---|---|---|---|---|---|
| base | ... | bf16 | ... | ... | ... | ... | ... | PASS |

## 15. 优化 checklist

- [ ] 已有 PyTorch reference
- [ ] CRUN correctness 通过
- [ ] baseline 已记录
- [ ] build_context 已查看
- [ ] NPU trace 已查看
- [ ] 瓶颈是 compute / memory / sync / scheduling 中哪类
- [ ] tile 实验只改变一个变量
- [ ] OCM 未超限
- [ ] 输出无写冲突
- [ ] 多 core 无长期空闲
- [ ] 多 chip wait 有配对 update
- [ ] cache 行为已确认
- [ ] 低精度结果可接受
- [ ] 多 shape 复测

## 16. 小结

AXDSL 性能优化不是猜测，而是把问题拆成四个可观测对象：

```text
Graph：算子与融合
Memory：DDR / OCM / traffic
Schedule：jobs、overlap、bubble
Parallel：core / chip idle 与同步
```

工具链提供了 graph、build context、NPU trace 和 cycle trace。善用这些报告，才能把“感觉更快”变成可解释、可复现、可传播的优化结论。
