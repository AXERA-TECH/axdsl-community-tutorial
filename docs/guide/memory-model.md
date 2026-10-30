# 内存模型与数据搬运

AXDSL 的重要特点是**显式表达数据位置和搬运**。理解本章后，你可以回答：

- 输入输出在哪里？
- tile 计算发生在哪里？
- 什么时候需要 Shared OCM？
- 多芯片之间如何互访？
- 为什么 load / store / move 会影响性能？

## 1. AX8860 内存层级

AX8860 可用多个 chip 协同。每 chip 内包含：

- 4 个 core
- 每个 core 的私有 OCM
- chip 内所有 core 可访问的 Shared OCM
- chip 对应的 DDR
- chip 间 Neulink 通信模块

简化结构：

```text
Chip 0
├── Core 0 ── private OCM
├── Core 1 ── private OCM
├── Core 2 ── private OCM
├── Core 3 ── private OCM
├── Shared OCM
└── DDR 0
        ▲
        │ Neulink
        ▼
Chip 1
├── Core 0 ── private OCM
├── Core 1 ── private OCM
├── Core 2 ── private OCM
├── Core 3 ── private OCM
├── Shared OCM
└── DDR 1
```

::: tip
具体容量、带宽和地址映射以目标工具链与硬件文档为准。教程重点是建立编程模型，不承诺具体硬件参数。
:::

## 2. MemoryType

| MemoryType | 作用域 | 典型用途 |
|---|---|---|
| `MemoryType.DDR` | chip 对应外部内存 | Kernel 输入、输出、大 buffer |
| `MemoryType.OCM` | core 私有 | 当前 core 的 tile、局部临时 tensor |
| `MemoryType.SHARED_OCM` | chip 内 core 共享 | 多 core 中间结果、协作 buffer |
| `MemoryType.SIGNAL` | 同步机制 | core / chip 间状态同步 |

### DDR

特点：

- 容量大
- 相对 OCM 带宽低、延迟高
- 适合公开输入输出和大 tensor
- 频繁读写中间结果会带来明显 traffic

### OCM

特点：

- core 私有
- 适合当前 program 的工作 tile
- 容量有限，需要控制 tile 和临时变量
- 是计算引擎附近的高效工作区

### Shared OCM

特点：

- chip 内多个 core 可见
- 适合交换中间结果
- 需要考虑写冲突和同步
- 不等于跨 chip 全局内存

### Signal

用于表达：

- 某块数据已经写入
- 某阶段已经完成
- core / chip 间等待条件
- producer-consumer 依赖

## 3. 标准数据流

最常见 Kernel 数据流：

```text
DDR input slice
   ↓ ax.dsl.load
OCM tile
   ↓ compute
OCM result
   ↓ ax.dsl.store
DDR output slice
```

示例：

```python
x_tile = ax.dsl.load(x[start:stop])
y = ax.dsl.silu(x_tile)
ax.dsl.store(x=y, r=output[start:stop])
```

这种模式的优势：

- 每次 load 的数据量明确
- 中间计算不需要反复访问 DDR
- output 写回位置明确
- traffic 可以在性能报告中核对

## 4. 数据搬运 API

### 4.1 `ax.dsl.load`

把 DDR tensor 或切片载入计算可用内存。

```python
x_tile = ax.dsl.load(x[m0:m1, :])
```

建议：

- load 粒度与 tile 粒度一致
- 避免同一数据在多个 program 中重复 load
- 首次开发时显式 load，优化期再分析是否可合并

### 4.2 `ax.dsl.store`

把计算结果写回输出。

```python
ax.dsl.store(x=result, r=output[m0:m1, n0:n1])
```

确保每个输出区域只由明确 owner 写入，避免 race。

### 4.3 `ax.dsl.move`

在内存层级或 tensor 之间移动数据，常用于把结果写入 caller 指定目标。

```python
ax.dsl.move(result, r=target)
```

也可用于 OCM 与 Shared OCM 之间的数据传递，具体约束以 API 文档为准。

### 4.4 `ax.dsl.dynamic_ld`

按运行期 offset / shape 读取。

适合：

- dynamic token
- dynamic batch
- KV cache 位置
- 循环中运行期决定的 tile

示意：

```python
tile = ax.dsl.dynamic_ld(
    x,
    read_offset,
    tile_shape,
    dim=(0,),
    output_shape=(1, M, N),
)
```

### 4.5 `ax.dsl.dynamic_st`

把结果写到运行期 offset / shape。

```python
ax.dsl.dynamic_st(
    result,
    zero_offset,
    write_offset,
    tile_shape,
    dim=(0,),
    output_shape=output.shape,
    r=output,
)
```

## 5. Tensor 的 memory type

`DslTensor` 包含 memory type 信息。Kernel 输入通常来自 caller 的 Torch tensor，被视为 DDR tensor；`load` 后产生的 tile 会在局部内存上参与计算。

`create_tensor` 可以显式创建中间 tensor：

```python
temp = ax.dsl.create_tensor(shape=x.shape, dtype=x.dtype)
ax.dsl.add(x, y, r=temp)
```

如需指定 memory type，应查看当前工具链的 `create_tensor` / 相关 API 签名。

## 6. 单 core 数据流示例

```python
@ax.builder(run_mode=ax.RunMode.PYRUN)
def relu_kernel(x, output, tile_size):
    start = ax.program_id(0) * tile_size
    stop = min(start + tile_size, x.shape[0])

    x_tile = ax.dsl.load(x[start:stop])
    zero = ax.dsl.zeros(shape=x_tile.shape, dtype=x_tile.dtype)
    result = ax.dsl.maximum(x_tile, zero)
    ax.dsl.store(x=result, r=output[start:stop])
    return output
```

数据流：

```text
x DDR slice → x_tile OCM
x_tile, zero → maximum OCM
result → output DDR slice
```

## 7. 多 core 数据流

多 core 常见模式：

```text
每个 core:
  load 自己的 DDR slice
  写入自己的 OCM tile
  计算
  写回自己的 DDR output slice
```

如果 core 之间需要交换数据：

```text
Core 0 OCM
   ↓ move / write
Shared OCM
   ↓ read / move
Core 1 OCM
```

或者：

```text
Core 0 计算结果
   ↓ store / move
Shared OCM
   ↓ signal
Core 1 等待 signal 后读取
```

必须通过 signal 或同步 API 保证可见性，不能只依赖 Python 代码顺序。

## 8. 多芯片数据流

每个 chip 有自己的 DDR。AX8860 chip 间通过 Neulink 访问或通信。

### 邻居发送

```text
Chip 0 DDR block → Neulink → Chip 1 DDR/destination
```

### Ring all-gather

以 4 chip 为例：

```text
Step 0: 每 chip 把本地块发给右邻居
Step 1: 每 chip 转发收到的前一块
Step 2: 继续转发
Step 3: 每 chip 拥有完整数据
```

详见 [Ring All-Gather 教程](/tutorials/ring-all-gather)。

## 9. Tile 设计

### 9.1 tile 与 OCM

每个 program 的工作集包括：

- 输入 tile
- 权重 tile
- 输出 tile
- 中间临时 tensor

估算时不能只看输入大小。例如 GEMM tile 需要：

```text
A tile + B tile + C tile + 临时 buffer
```

### 9.2 tile 与 DDR traffic

更大 tile 通常可以提升数据复用，例如 GEMM 中：

```text
A tile 复用 N 方向多次
B tile 复用 M 方向多次
```

但更大 tile 会增加 OCM 压力并降低并行度。应通过性能报告比较。

### 9.3 tile 与尾块

非整除 shape 需要处理：

```python
m1 = min((pid_m + 1) * TILE_M, M)
```

尾块可能降低引擎利用率。如果某个 shape长期使用，可以为其选择专用 tile 策略。

## 10. 常见内存错误

### OCM 超限

表现：

- 编译失败
- 提示 memory / OCM 超限
- 某些 tile 组合不可调度

处理：

- 减小 tile
- 减少临时 tensor
- 拆分循环
- 复用 buffer
- 分析 build context

### 重复 load

表现：

- DDR traffic 高
- 性能报告显示搬运占比大

处理：

- 提高 tile 复用
- 调整 grid 映射
- 将中间结果留在 OCM
- 合并算子

### 输出写冲突

表现：

- 多 program 写同一区域
- 非确定错误

处理：

- 明确每个 program 的输出 owner
- index 类写入确保语义允许
- 使用同步或原子/专用 API 前确认支持

### 未写满输出

表现：

- PYRUN / CRUN 输出含未定义值
- assert 偶发失败

处理：

- 检查尾块
- 检查 grid 覆盖范围
- 初始化输出为 0 仅作为调试手段
- 用小 shape 打印 index / shape

### 同步缺失

表现：

- 读取到旧数据
- 多核偶发失败

处理：

- 为 shared 数据增加 signal
- 明确 producer / consumer
- 检查 signal value 是否重复或回绕

## 11. 性能直觉

一个 Kernel 的近似执行时间可以看作：

```text
计算时间
+ DDR / Neulink 搬运时间
- 可 overlap 的搬运与计算时间
+ 调度与同步开销
+ 空闲 / bubble
```

优化方向：

- 减少总 traffic
- 让数据尽量留在 OCM
- 提高 tile 复用
- 让 DMA 与计算 overlap
- 减少 core 空闲
- 减少同步等待
- 降低通信粒度过小带来的开销

## 12. 小结

| 问题 | AXDSL 表达 |
|---|---|
| 从 DDR 取 tile | `load` |
| 写回 DDR | `store` |
| 移动到指定目标 | `move` |
| 运行期偏移读取 | `dynamic_ld` |
| 运行期偏移写入 | `dynamic_st` |
| core 内局部工作区 | OCM |
| core 间共享 | Shared OCM + signal |
| chip 间通信 | Neulink / shmem API |

理解这些 API 后，AXDSL Kernel 的性能热点就能被明确拆解为：**tile 设计、搬运次数、并行度和同步等待**。
