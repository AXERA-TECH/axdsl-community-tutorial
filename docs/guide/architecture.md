# 总体架构与工作原理

理解 AXDSL 的关键是分清两层：

1. **Python 层**：开发者写函数、切片、循环、调用 AXDSL API。
2. **Graph IR 层**：builder 记录这些 API 调用，生成后端可优化的计算图。

AXDSL 的 Python 代码不是普通 Python 函数在 NPU 上逐行解释执行，而是被 trace 成 IR 后再执行或编译。

## 1. 在工具链中的位置

```text
开发者
  │
  ▼
Python Kernel + Torch runtime tensors + compile-time constants
  │
  ▼
@ax.builder / kernel[grid](...)
  │  记录 API 调用与子图
  ▼
AXDSL Graph IR
  │  算子融合、布局与调度变换、tile / job 生成
  ▼
AX8860 编译器
  │
  ▼
AXModel / 编译产物
  │
  ├── PYRUN：Python reference 执行
  ├── CRUN：编译后 simulator 执行
  └── BRUN：设备 / AXCL 后端执行
```

这张图可以概括为三个阶段：

| 阶段 | 关注点 | 开发者可见内容 |
|---|---|---|
| Record | Python trace 成 Graph | 源代码、输入输出、tensor 元数据 |
| Compile | 图优化、算子映射、job 调度 | graph dump、build context、trace |
| Run | reference / simulator / device 执行 | 输出、校验结果、cycle 数据 |

## 2. 核心对象

### 2.1 Recording Context

`ax.init` 初始化当前上下文，主要保存：

- chip 数
- 当前 active kernel
- 每个 chip 的输入输出映射
- grid 配置
- tensor 注册信息
- 编译与运行选项

```python
ax.init(chip_num=1, core_num=1)
```

重新初始化会重置上下文配置。

### 2.2 KernelBuilder

`@ax.builder` 返回 `KernelBuilder` 对象。它负责：

- 注册 Python kernel 函数
- 保存 run mode、check、cache 等配置
- 解析 launch grid
- 处理输入输出
- 记录 Graph
- 执行 PYRUN / CRUN / BRUN
- 管理编译缓存
- 导出调试与性能产物

```python
@ax.builder(
    run_mode=ax.RunMode.CRUN,
    enable_cache=False,
    check=ax.Check.ALL_CLOSE,
)
def kernel(...):
    ...
```

### 2.3 Grid 与 Program

Grid 是逻辑 program 空间，不一定直接等价于某个固定硬件线程模型。Kernel 作者需要定义 program 坐标与 tensor 区域的映射。

```python
@ax.builder
def kernel(x, output):
    i = ax.program_id(0)
    ...
```

常用 API：

- `ax.program_id(axis)`：当前坐标
- `ax.num_programs(axis)`：当前 axis 的 program 数

### 2.4 DslTensor

`DslTensor` 描述：

- shape
- dtype
- memory type
- name / producer 关系

Kernel 内不能读取 tensor 的运行期值。可以读取 shape、dtype 等元数据。

### 2.5 Graph IR

AXDSL Graph IR 由节点和 tensor 组成：

- 普通计算节点：`AxAdd`、`AxMatMul`、`AxReduceSum` 等
- 数据移动节点：load / store / move
- 控制流节点：`AxIf`、`AxWhileLoop`、`AxRange`、`AxSwitch`
- 通信节点：Neulink / shared memory 相关节点
- 子图：控制流 branch / loop body

## 3. 一次 launch 的执行顺序

以：

```python
kernel[(4, 4), 1](x, y, output, tile=(64, 64))
```

为例，执行顺序大致为：

```text
1. __getitem__((4, 4), 1)
   解析 grid_shape 和 core_num，准备 context

2. __call__(x, y, output, tile)
   根据 run_mode 分发

3. process_io
   识别输入输出，Torch tensor 转为 DSL tensor

4. gen_graph
   trace Python 函数，记录算子与子图

5. fuse_ops / transform
   做图级优化与模式匹配

6. compile / simulate / execute
   根据 PYRUN / CRUN / BRUN 分支处理
```

### 多芯片差异

当 `chip_num > 1` 时，通常需要对每个 chip 调用一次同一个 Kernel builder：

```python
for chip_id in range(CHIP_NUM):
    kernel[1](x_chip[chip_id], output_chip[chip_id], signal, chip_id, CHIP_NUM)
```

AXDSL 会依次记录每个 chip 的图，所有 chip 记录完成后再统一执行多芯片仿真。

## 4. Record 语义与 Python 语义

AXDSL 支持用 Python 组织代码，但要区分三类行为。

### 4.1 编译期 Python 逻辑

普通 Python 值可以直接参与逻辑：

```python
if tile_size > 128:
    ...
for i in range(0, n, tile_size):
    ...
```

这类逻辑在 trace 时确定，会影响生成的 IR。

### 4.2 运行期 tensor 逻辑

tensor 值必须通过 DSL API 计算：

```python
result = ax.dsl.add(x_tile, y_tile)
```

不能读取：

```python
if x_tile.numpy().sum() > 0:  # 不允许
```

### 4.3 结构化运行期控制流

如果条件或循环依赖运行期 tensor，需要使用：

```python
ax.dsl.if_op(...)
ax.dsl.while_loop(...)
ax.dsl.range(...)
ax.dsl.switch(...)
```

这些 API 会把 body trace 成子图，并在运行期决定执行路径。

## 5. 图优化与算子映射

AXDSL 生成 Graph 后，后端可以做多种变换，例如：

- 常量折叠
- elementwise fusion
- FMA / reduce 模式匹配
- layout 变换
- 子图调度
- DMA 与计算 overlap
- job 划分
- 控制流展开或 lane cloning

这些优化不会改变用户可见语义，但可能影响精度、中间 tensor 和性能。因此建议：

- 开发期保留 reference 对照
- 性能调优时打开 graph dump 和 trace
- 修改后同时跑 correctness case 和 performance case

## 6. 输入输出模型

AXDSL Kernel 通常采用 caller-allocated output：

```python
def kernel(x, y, output, tile_size):
    ...
    return output
```

原因：

1. Kernel 不需要隐式分配返回值
2. 输出地址可由 caller 控制
3. 更容易集成到推理引擎 custom op
4. 多核 / 多芯片时每个 program 写自己的区域
5. 可以显式表达 partial output 写回

多个输出时：

```python
def kernel(x, out_a, out_b):
    ...
    return out_a, out_b
```

## 7. 为什么输出要 return

返回值帮助 builder 识别输出契约：

- 输入和输出顺序
- 输出数量
- 输出 tensor 元数据
- PyRun 与 Crun 对比目标

即使 output 已经通过参数传入，也建议保留显式 return。

## 8. 三种运行视角

### PYRUN

用 Python reference implementation 执行记录的算子。

适合：

- 第一时间验证算法语义
- 小 shape 快速迭代
- 不依赖后端编译的问题定位

### CRUN

走后端编译和 simulator。

适合：

- 验证 Graph IR 到后端映射是否正确
- 检查 dtype / shape / 内存限制
- 获取估算 cycle 与调度信息
- 开启输出或逐层 check

### BRUN

运行编译后的 AXModel / 设备后端。

适合：

- 真实设备验证
- 与 AXCL / runtime 集成
- 最终性能测试

详细对比见 [执行、验证与调试](/guide/execution)。

## 9. AXDSL 的设计取舍

### 为什么显式 load / store

NPU 性能高度依赖数据搬运。如果完全依赖隐式搬运，用户很难回答：

- tile 是否留在 OCM？
- 中间结果是否写回 DDR？
- 是否重复搬运？
- 通信和计算是否重叠？
- OCM 是否超限？

显式 API 让这些问题可见、可分析、可优化。

### 为什么限制普通 Python 赋值

Graph IR 需要 tensor producer / consumer 关系。隐式 Python 赋值容易引入 alias 和副作用，难以可靠推导输出。`r=output_slice` 明确表达写目标。

### 为什么控制流是结构化 API

任意 Python 控制流很难完整映射到后端。结构化子图让条件、循环和分支具有清晰输入输出契约，方便 PyRun、编译器和 simulator 保持一致语义。

### 为什么保留 compile-time constant

tile size、grid、layout 等参数常常决定生成的代码形状。作为编译期常量可以让后端做更激进优化；代价是不同常量组合可能生成不同编译产物。

## 10. 小结

AXDSL 的核心工作流是：

```text
写 Python Kernel
  ↓
用 builder 记录成 Graph IR
  ↓
后端优化并编译
  ↓
PYRUN / CRUN / BRUN 验证执行
  ↓
trace 与性能报告指导下一轮优化
```

理解这个链路后，后续章节中的 grid、内存、控制流和通信 API 都可以看作在丰富 Graph IR 的表达能力。
