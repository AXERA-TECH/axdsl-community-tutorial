# 为什么选择 AXDSL

AXDSL 是 AX8860 NPU 工具链中的 Python 自定义 Kernel DSL。它不是“把 Python 直接跑在 NPU 上”的通用 Python 编译器，而是一个**面向 NPU Kernel 的记录式 DSL**：开发者用 Python 描述算法、tile 划分、数据搬运和并行协作，`ax.builder` 把这段函数记录成 AXDSL Graph IR，再交给后端完成图优化、算子映射、任务调度和编译。

一句话概括：

> 用 Python 写 Kernel 的开发体验，保留 NPU 需要的显式控制与性能可观测性。

## 传统自定义算子开发的痛点

在 NPU 上写自定义算子时，开发者通常会遇到四类问题。

### 1. 表达门槛高

底层 Kernel 开发要求同时理解：

- 算法语义
- 数据布局
- 内存层级
- DMA / 计算引擎调度
- 多核同步
- 精度与量化

这些细节都直接暴露在实现里，算法工程师很难快速把想法变成可运行版本。

### 2. 图算子拼接性能受限

通过已有图算子组合新功能，虽然入门简单，但经常出现：

- 中间结果反复写回 DDR
- 小算子过多，调度开销高
- 难以做跨算子融合
- 特定 shape 下无法选择更优 tile
- 表达动态控制流困难

最终性能往往取决于图优化器能否识别模式，而不是开发者真正知道的硬件友好写法。

### 3. 动态执行难表达

LLM、检测、检索、稀疏计算等场景经常需要：

- 根据运行期数据选择分支
- 动态 token 数 / batch / KV 长度
- 提前终止循环
- 按索引读写
- 多阶段循环并携带状态

传统静态图模型很难自然表达这些逻辑。

### 4. 并行和通信复杂

AX8860 的能力来自多个层次：

- 每 chip 多个 core
- core 私有 OCM
- chip 内 Shared OCM
- 每 chip DDR
- chip 间 Neulink

如果没有统一抽象，多核和多芯片代码会快速演变成难以维护的底层同步代码。

## AXDSL 的解决思路

AXDSL 把开发过程拆成五个可控层次：

```text
算法语义          Python / AXDSL 算子 API
任务划分          Program Grid
数据位置          DDR / OCM / Shared OCM
数据搬运          load / store / move / dynamic_ld / dynamic_st
协同同步          signal / shmem / Neulink API
```

这带来三个重要变化。

### 1. Kernel 结构更容易读

一个 tile 化 Kernel 可以直接读出数据流：

```python
x_tile = ax.dsl.load(x[start:stop])       # DDR → OCM
y_tile = ax.dsl.load(y[start:stop])
result = ax.dsl.add(x_tile, y_tile)       # OCM 上计算
ax.dsl.store(x=result, r=output[start:stop]) # OCM → DDR
```

读者能明确看到数据从哪里来、在哪里算、写回哪里。

### 2. 性能关键决策不被黑盒隐藏

AXDSL 不承诺“随便写就自动最优”。它把最重要的硬件决策暴露为可理解、可调整的 API：

- tile 大小
- grid 形状
- core 数
- load / store 时机
- 中间 tensor 所在内存
- 多核通信粒度
- 控制流展开程度

这更适合高性能 Kernel 工程。

### 3. 正确性与性能工具同链路打通

开发流程内建：

- PYRUN Python reference
- CRUN 编译仿真
- BRUN 设备运行
- 输出 / 逐层正确性检查
- graph 与编译产物 dump
- NPU trace 与 cycle trace
- HTML 性能报告

从第一个版本到最后调优，不需要切换到完全割裂的工具链。

## 与其他开发方式对比

| 维度 | 手写底层 Kernel | 组合已有图算子 | AXDSL |
|---|---|---|---|
| 入门难度 | 高 | 低 | 中低 |
| 表达能力 | 强 | 受已有算子限制 | 较强 |
| Python 体验 | 无 | 有 | 有 |
| 数据搬运控制 | 强 | 弱 | 强 |
| tile 与并行策略 | 手工 | 多依赖通用优化 | 显式可写 |
| 控制流 | 取决于底层工具 | 常受限 | 结构化 API |
| 多核 / 多芯片 | 手工复杂 | 依赖框架 | DSL API 支持 |
| 性能可解释性 | 强但成本高 | 弱 | 较强 |
| 适合场景 | 极致底层优化 | 简单扩展 | 高性能自定义 Kernel |

AXDSL 的目标不是替代所有底层开发，而是让绝大多数自定义算子能在更高抽象层完成，并让少数确实需要底层介入的问题有清晰下降路径。

## 核心技术优势

### 1. Python 记录式开发

```python
@ax.builder(run_mode=ax.RunMode.PYRUN)
def kernel(...):
    ...
    return output
```

`@ax.builder` 标记 Kernel，`kernel[grid](...)` 启动录制和执行。Python 参数、切片和循环可用于组织逻辑，AXDSL 算子记录计算图。

### 2. 显式内存模型

AXDSL 不把 DDR 与 OCM 的区别完全隐藏。开发者可以清楚表达：

- 输入输出在 DDR
- 计算局部 tile 在 OCM
- 跨 core 共享数据在 Shared OCM
- 同步状态在 Signal

对高性能 Kernel 来说，这是性能可控的关键。

### 3. Program Grid 并行模型

```python
kernel[(grid_m, grid_n), core_num](...)
```

Grid 是逻辑 program 空间。Kernel 内用 `ax.program_id(axis)` 获取坐标，自行定义该 program 拥有哪些 tensor 区域。它既能表达一维切分，也能表达二维 tile 或更复杂的任务映射。

### 4. 结构化动态控制流

AXDSL 提供：

- `if_op`
- `while_loop`
- `range`
- `switch`
- `dynamic_ld`
- `dynamic_st`

它们会被记录为包含子图的 IR 节点，而不是简单在 Python 编译期展开，因此能表达运行期选择和循环。

### 5. 多芯片协同抽象

AX8860 支持 chip 间通信。AXDSL 提供：

- `shmem_put`
- `signal_update`
- `signal_wait`
- `shmem_signal_update`
- shared memory tensor / signal 创建

这让 ring all-gather、多 chip GEMM 等模式可以以 Kernel 语义表达。

### 6. 工程化工具

- `enable_cache`：复用编译产物
- `dump_dsl_perf`：导出性能分析材料
- `DUMP_NPU_TRACE=1`：查看 scheduled jobs 与 overlap
- `DEBUG_SIM_CYCLE_TRACE=1`：查看仿真事件时序
- `build_context.json`：查看 jobs、traffic、内存与估算 cycle

## 能力矩阵

| 类别 | 代表 API | 说明 |
|---|---|---|
| Kernel 定义 | `ax.builder` | Python 函数变为可启动 Kernel |
| 初始化 | `ax.init` | 配置参与 chip 数 |
| Program | `program_id`, `num_programs` | 获取逻辑任务坐标与范围 |
| 内存 | `load`, `store`, `move` | DDR / OCM / Shared OCM 搬运 |
| Elementwise | `add`, `sub`, `mul`, `div` | 基础逐元素计算 |
| Activation | `sigmoid`, `gelu`, `silu`, `softplus` | 常见激活函数 |
| Reduction | `sum`, `max`, `min`, `argmax`, `cumsum` | 维度归约 |
| GEMM / Conv | `matmul`, `conv` | 核心 dense 计算 |
| Shape | `reshape`, `transpose`, `concat`, `expand`, `pad` | 布局和形状变换 |
| Index | `gather`, `scatter`, `index_select`, `topk`, `nonzero` | 动态索引 |
| Control Flow | `if_op`, `while_loop`, `range`, `switch` | 运行期结构化控制流 |
| Multi-core | `sync_all`, signal, Shared OCM | chip 内 core 协作 |
| Multi-chip | `shmem_put`, `signal_wait` | Neulink 通信 |
| Validation | `check`, `check_mode`, tolerance | PYRUN / CRUN 对比 |
| Performance | trace, perf artifacts | cycle、traffic、利用率分析 |

## AXDSL 不是什么

为了避免社区误解，推广时建议明确以下边界：

1. **不是通用 Python 加速器**  
   Kernel 内支持的 Python 语法和库有明确边界，不是任意 Python 都能映射到 NPU。

2. **不是完全隐式自动优化**  
   tile、grid、内存位置和通信方式仍由开发者设计；AXDSL 提供表达能力和分析工具。

3. **不是所有算子都支持所有 dtype / shape / backend**  
   发布前应检查目标工具链的 API Reference 和支持矩阵。

4. **不是只能写小算子**  
   它也可以表达 tile 循环、动态执行、多核通信和多芯片协作。

## 小结

AXDSL 的价值可以概括为：

- **上手快**：Python Kernel 写法，示例清晰。
- **控制强**：显式表达 tile、内存和并行。
- **能力全**：覆盖控制流、多核、多芯片和动态读写。
- **可验证**：PYRUN / CRUN / BRUN 与逐层检查。
- **可调优**：编译产物、trace 和性能报告联动。
- **适合社区传播**：从 Vector Add 到 All-Gather 都能用同一套模型讲解。
