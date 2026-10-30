# 执行、验证与调试

本章给出一套推荐的 correctness-first 工作流：先保证 reference 与 DSL 语义一致，再验证编译执行，最后进入性能优化。

## 1. 三种 Run Mode

| Mode | 执行方式 | 适合阶段 |
|---|---|---|
| `RunMode.PYRUN` | Python reference implementation | 快速验证语义，不需要后端编译 |
| `RunMode.CRUN` | 后端编译 + simulator | 验证后端映射、精度与调度 |
| `RunMode.BRUN` | 编译 AXModel 并走设备 / AXCL runtime | 真实设备运行与集成 |

默认 run mode 与工具链版本有关；教程和示例中建议显式声明，避免升级后行为变化。

## 2. PYRUN：语义优先

```python
@ax.builder(run_mode=ax.RunMode.PYRUN)
def kernel(...):
    ...
```

适用：

- 第一版 Kernel
- 小 shape 快速调试
- Python 语义确认
- control flow API 语义确认

局限：

- 不代表后端编译一定成功
- 不反映真实性能
- 后端 dtype / shape 限制仍需 CRUN 验证

## 3. CRUN：编译与仿真

```python
@ax.builder(
    run_mode=ax.RunMode.CRUN,
    enable_cache=False,
    check=ax.Check.ALL_CLOSE,
)
def kernel(...):
    ...
```

适用：

- 验证 graph 到后端映射
- 检查 OCM、job、调度约束
- 与 PYRUN 比较输出
- 获取 simulator cycle 和 trace

注意：

- `check` 与 `enable_cache=True` 不能同时使用
- 多芯片与性能 dump 的支持范围可能受版本限制
- 编译失败时应优先缩小 shape 并定位最小复现

## 4. BRUN：设备运行

BRUN 通常要求：

- 单 chip（以当前源码限制为准）
- 启用 cache
- 初始化 AXCL session
- 输入输出位于对应 runtime device

典型流程：

```python
ax.init(chip_num=1)
ax.init_session(...)
kernel[grid](device_inputs, device_outputs)
ax.synchronize_stream()
ax.finalize()
```

具体 session API 与设备 tensor 构造方式以工具链文档为准。

## 5. 正确性检查配置

示例：

```python
@ax.builder(
    run_mode=ax.RunMode.CRUN,
    check=ax.Check.ALL_CLOSE,
    check_mode=ax.CheckMode.CHECK_PER_LAYER,
    check_rtol=1e-5,
    check_atol=1e-5,
)
def kernel(...):
    ...
```

| 参数 | 说明 |
|---|---|
| `check` | 检查策略 |
| `check_mode` | 只检查最终输出或逐层输出 |
| `check_rtol` | 相对误差 |
| `check_atol` | 绝对误差 |
| `check_cosine_similarity` | 余弦相似度阈值 |
| `check_tensor_black_list` | 排除某些中间 tensor |

### 常用策略

- `ALL_CLOSE`：适合大多数 elementwise / GEMM 输出
- `COSINE_SIMILARITY`：适合整体方向重要、幅度可容忍的向量
- `CHECK_OUTPUT`：定位最终输出
- `CHECK_PER_LAYER`：定位第一个出错中间层

## 6. 推荐开发流程

```text
1. 写 PyTorch reference
2. 设计 shape / tile / grid
3. 用最小 shape 实现 AXDSL
4. PYRUN 验证
5. 增加非整除 shape、边界 case、随机数据
6. CRUN 编译验证
7. 开启 output / per-layer check
8. 扩展多 core
9. 扩展多 chip
10. 打开 trace 和性能报告
11. 迭代优化
```

### 为什么先写 reference

reference 提供三方对照：

```text
PyTorch reference
   vs AXDSL PYRUN
   vs AXDSL CRUN
```

这样可以区分：

- 算法写错：PyTorch 与 PYRUN 不一致
- 后端映射或精度问题：PYRUN 与 CRUN 不一致
- 集成问题：CRUN 与 BRUN 不一致

## 7. 测试 case 设计

| Case | 目的 |
|---|---|
| 1 × tile | 最小语义验证 |
| 多个完整 tile | grid 映射 |
| 非整除尾块 | 边界处理 |
| 极小 shape | 空切片 / 特殊维度 |
| 不同 dtype | 后端支持与精度 |
| 随机值 | 避免只对常数成立 |
| 全 0 / 全 1 | 归一化、scale、除零策略 |
| 大值 / 小值 | 数值范围 |
| 多次调用 | 状态污染与缓存 |

## 8. 打印与定位

### device_print

```python
ax.dsl.device_print(...)
```

适合查看设备端调试信息，但可能影响调度与性能，不建议在最终性能测试中保留。

### 缩小复现

推荐步骤：

1. 固定随机种子
2. 使用最小 shape
3. 只保留一个 program
4. 只保留一个算子
5. 打开 `CHECK_PER_LAYER`
6. 恢复一层，验证一次

### 对比中间结果

把 reference 中间结果与 AXDSL 中间输出对比：

```python
def reference_stage1(x):
    return ...
```

不要只对比最终输出。对 fusion / reduce / softmax 类算子尤其重要。

## 9. 调试产物

常用配置：

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

环境变量：

```bash
DUMP_NPU_TRACE=1 \
DEBUG_SIM_CYCLE_TRACE=1 \
python3 my_kernel.py
```

| 产物 | 内容 | 用途 |
|---|---|---|
| source snapshot | DSL 源码 | 对齐版本 |
| graph snapshot | raw / fused graph | 查看算子与融合 |
| build context | jobs、traffic、内存、估算 cycle | 快速定位资源 |
| NPU trace | scheduled jobs、overlap、利用率 | 分析流水 |
| simulator cycle trace | 运行期事件 | 分析动态控制流 |
| HTML report | 可视化报告 | 汇总展示 |

## 10. 常见错误分类

### 语法 / tracing 错误

典型原因：

- 不支持的 Python 写法
- 读取运行期 tensor 值
- 使用降 rank 索引
- control flow body 返回值不符合契约

处理：

- 阅读编程模型中的语法限制
- 将普通分支改为 `if_op`
- 将普通循环改为 `range` / `while_loop`

### Shape / slice 错误

典型原因：

- grid 与切片映射不一致
- 尾块未处理
- 输出未写满
- tile 超过 shape

处理：

- 打印 grid、program_id、slice 范围
- 先使用 1 个 program
- 使用非整除 case

### dtype / 精度错误

典型原因：

- 后端不支持该 dtype
- reference 与 DSL dtype 不一致
- low precision 容差过严
- reduce / softmax 数值稳定差异

处理：

- fp32 先验证语义
- 再切换 bf16 / fp16 / int8
- 根据任务选择 tolerance

### 内存错误

典型原因：

- tile 或临时 tensor 超过 OCM
- 重复保留中间结果
- shared tensor 冲突

处理：

- 减小 tile
- 拆分循环
- 查看 build context

### 并行错误

典型原因：

- signal 顺序错误
- 读写冲突
- producer/consumer 缺同步
- 多 chip 输出切片不一致

处理：

- 画数据流图
- 标出每块数据的 owner
- 为每次 remote write 配对 signal
- 从 1 core / 1 chip 逐步扩展

### CRUN 与 PYRUN 差异

可能原因：

- 后端不支持某个组合
- 中间 dtype 提升不同
- 图融合改变计算顺序
- control flow 子图契约不一致
- 内存 alias 未按规则表达

处理：

- 开启 per-layer check
- 暂时关闭融合或减小 case（如果工具支持）
- 保留最小复现提交给工具链团队

## 11. 最小调试模板

```python
import torch
import axnn.dsl as ax


def reference(x, y):
    return x + y


@ax.builder(run_mode=ax.RunMode.CRUN, enable_cache=False,
            check=ax.Check.ALL_CLOSE, check_mode=ax.CheckMode.CHECK_PER_LAYER,
            check_rtol=1e-3, check_atol=1e-3)
def kernel(x, y, output, tile_size):
    start = ax.program_id(0) * tile_size
    stop = min(start + tile_size, x.shape[0])
    a = ax.dsl.load(x[start:stop])
    b = ax.dsl.load(y[start:stop])
    out = ax.dsl.add(a, b)
    ax.dsl.store(x=out, r=output[start:stop])
    return output


def test():
    torch.manual_seed(0)
    x = torch.randn(257, dtype=torch.bfloat16)
    y = torch.randn_like(x)
    got = torch.empty_like(x)
    ax.init(chip_num=1, core_num=1)
    kernel[3](x, y, got, 128)
    torch.testing.assert_close(reference(x, y), got, atol=1e-2, rtol=1e-2)
```

## 12. 小结

| 问题 | 推荐定位方式 |
|---|---|
| 算法语义错 | PyTorch reference + PYRUN |
| 输出偶发错 | 尾块、未写满、随机数据 |
| 后端映射错 | CRUN + CHECK_OUTPUT |
| 中间层错 | CHECK_PER_LAYER |
| 资源超限 | build context |
| 调度 / overlap 差 | NPU trace |
| 动态循环行为异常 | simulator cycle trace |
| 多核同步错 | 数据流图 + signal 配对 |
