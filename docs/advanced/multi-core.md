# 多核并行编程

AX8860 每个 chip 内有 4 个 core。AXDSL 的多核编程入口是 **Program Grid**：每个 program 对应一段被记录的 Kernel 逻辑，通过 `program_id` 获得自己的坐标，再由 Kernel 作者决定它处理哪些数据。

## 1. 从单核到多核

单 program 版本：

```python
@ax.builder
def kernel(x, output):
    x_tile = ax.dsl.load(x)
    y = ax.dsl.silu(x_tile)
    ax.dsl.store(x=y, r=output)
    return output


kernel[1](x, output)
```

四个 program：

```python
kernel[4](x, output)
```

Kernel 内：

```python
core_id = ax.program_id(0)
```

`program_id` 是逻辑任务坐标。Kernel 代码必须根据它计算当前 program 拥有的 tensor 区域。

## 2. 常见数据划分

### 按一维元素切分

```python
BLOCK = (N + CORE_NUM - 1) // CORE_NUM
start = core_id * BLOCK
stop = min(start + BLOCK, N)
```

### 按行切分

```python
rows_per_core = M // CORE_NUM
m0 = core_id * rows_per_core
m1 = m0 + rows_per_core

x_tile = ax.dsl.load(x[m0:m1, :])
```

### 按二维 tile 切分

```python
pid_m = ax.program_id(0)
pid_n = ax.program_id(1)

m0 = pid_m * TILE_M
m1 = min(m0 + TILE_M, M)
n0 = pid_n * TILE_N
n1 = min(n0 + TILE_N, N)
```

启动：

```python
kernel[(M // TILE_M, N // TILE_N), CORE_NUM](...)
```

### 按任务维度切分

适合 batch、head、expert、sequence block：

```python
head = ax.program_id(0)
x_head = x[:, head:head+1, :]
```

## 3. core 数与 grid

可以显式指定：

```python
kernel[grid_shape, core_num](...)
```

例如：

```python
kernel[(8, 8), 4](...)
```

如果省略 `core_num`，默认值为 grid 元素数量。实际可用 core 数受 AX8860 每 chip core 数和后端调度限制。

## 4. 独立并行模式

如果 core 之间没有数据依赖，这是最理想的形式：

```text
core 0: input slice 0 → compute → output slice 0
core 1: input slice 1 → compute → output slice 1
core 2: input slice 2 → compute → output slice 2
core 3: input slice 3 → compute → output slice 3
```

特点：

- 无共享写冲突
- 无跨 core 依赖
- 负载由数据划分决定
- 通常最容易调试和调度

示例：

```python
@ax.builder(run_mode=ax.RunMode.PYRUN)
def rows_kernel(x, output, rows_per_core):
    core_id = ax.program_id(0)
    m0 = core_id * rows_per_core
    m1 = min(m0 + rows_per_core, x.shape[0])

    x_tile = ax.dsl.load(x[m0:m1, :])
    y = ax.dsl.mul(x_tile, x_tile)
    ax.dsl.store(x=y, r=output[m0:m1, :])
    return output
```

## 5. Shared OCM

core 私有 OCM 不能被其他 core 直接访问。要在多个 core 之间共享中间数据，需要使用 `MemoryType.SHARED_OCM`。

```python
from axnn.backend.base import MemoryType

shared = ax.dsl.create_tensor(
    shape=(M, K),
    dtype=x.dtype,
    memory_type=MemoryType.SHARED_OCM,
)
```

典型工作方式：

1. 每个 core 写入自己拥有的 shard
2. 发布该 shared tensor 的最新版本
3. 其他 core 的后续读取订阅该版本
4. 调度器根据 pub / sub 事件建立跨 core 依赖

## 6. 核间同步：pub / sub

当前交付 Kernel 中主要使用的核间同步机制是：

```python
from axnn.dsl import parallel as axd

published = axd.pub(shared)
subscribed = axd.sub(shared)
```

### `pub`

```python
axd.pub(shared)
```

含义：发布此前记录到 `shared` 上的写入。它会记录一个 tensor 版本事件，例如：

```text
shared@1
```

后续 `sub` 可以订阅这个版本。

### `sub`

```python
axd.sub(shared)
```

含义：让当前位置之后读取 `shared` 的 job 订阅最近可见的发布版本。调度器会把 consumer 依赖到对应 producer 上。

### 重要语义

- `pub` 与 `sub` 作用于同一个 DSL tensor
- `sub` 必须能看见当前作用域中之前的 `pub`
- `pub` 发布的是该点之前完成的写入
- `sub` 影响该点之后的读取
- 对 `SHARED_OCM` tensor，事件可以建立跨 core 依赖
- 对 core 私有 OCM tensor，事件不会让另一个 core 读到该 tensor
- `pub / sub` 是 **chip 内核间同步**，不是芯片间同步

## 7. Shard 交换示例

下面的模式来自当前源码测试和交付 Kernel 常用写法：每个 core 先把属于自己的输入 shard 写入同一个 `SHARED_OCM` tensor，再发布和订阅该 tensor。

```python
from axnn.backend.base import MemoryType
from axnn.dsl import parallel as axd

import axnn.dsl as ax


@ax.builder(run_mode=ax.RunMode.PYRUN)
def exchange_kernel(x, output, shard_size):
    core_id = ax.program_id(0)
    owned = slice(core_id * shard_size, (core_id + 1) * shard_size)

    shared = ax.dsl.create_tensor(
        shape=x.shape,
        dtype=x.dtype,
        memory_type=MemoryType.SHARED_OCM,
        name="exchange_shared",
    )

    # 当前 core 只写自己拥有的 shard。
    ax.dsl.load(x[owned], r=shared[owned])

    # 发布当前 core 已完成的写入，并订阅完整 shared tensor 的可见版本。
    axd.pub(shared)
    axd.sub(shared)

    # 后续读取 shared 时，调度器可以利用 pub/sub 依赖保证版本可见。
    value = ax.dsl.mul(shared, shared)

    # 输出同样按 owner 切分，避免多个 core 写同一区域。
    ax.dsl.store(x=value[owned], r=output[owned])
    return output
```

这个示例的重点是：

- `shared` 位于 `SHARED_OCM`
- 每个 core 写入自己的 shard
- `pub` 声明写入完成
- `sub` 声明后续读取需要可见版本
- 输出区域有唯一 owner

## 8. 常用 pub / sub 模式

### 多 producer 写不同 shard

```text
core 0 writes shared[0:16]
core 1 writes shared[16:32]
core 2 writes shared[32:48]
core 3 writes shared[48:64]

每个 core: pub(shared)
后续读取前: sub(shared)
```

### 分阶段计算

```text
stage 0:
  core 写 partial_0
  pub(partial_0)

stage 1:
  sub(partial_0)
  读取 partial_0
  计算 normalized
  pub(normalized)

stage 2:
  sub(normalized)
  读取 normalized
```

### 循环内版本发布

在 `range` / `while_loop` body 中，也可以对同一 shared state 成对使用 `pub` / `sub`。事件会在图内记录，后端调度时建立对应依赖。

## 9. 负载均衡

### 均匀 shape

尽量让每个 core 处理相同大小的 tile。

### 非均匀任务

例如 MoE token 数、变长 sequence、非零元素数量。可选方案：

- 静态近似划分
- batch 块再切细
- 通信 / 合并阶段拆分
- 专用 index / nonzero API
- 针对常见 shape 做特化

### 长尾 tile

尾块会降低利用率。可以：

- 对齐输入
- pad 后处理
- 小 tile 策略
- 将尾块合并到某个 core
- 为部署 shape 生成专用版本

## 10. 写冲突与数据所有权

多核调试第一原则：**每块输出必须有明确 owner**。

推荐画表：

| 输出区域 | Owner | 写入方式 |
|---|---|---|
| `output[0:128, :]` | core 0 | `store` |
| `output[128:256, :]` | core 1 | `store` |
| `shared[0:16]` | core 0 | `load(..., r=shared[0:16])` |
| `shared[16:32]` | core 1 | `load(..., r=shared[16:32])` |

如果 API 没有明确并发写语义，不要假设硬件会自动串行化。

## 11. 多核性能检查清单

- [ ] grid 与数据块一一对应
- [ ] 每个 core 工作量相近
- [ ] 尾块有明确 owner
- [ ] 输出没有写冲突
- [ ] shared tensor 每个 shard 有唯一 writer
- [ ] 读写跨 core 共享 tensor 时使用 pub / sub
- [ ] pub 与 sub 在相同作用域内配对
- [ ] 每个 core OCM 是否超限
- [ ] DDR traffic 是否随 core 数放大
- [ ] 性能报告中是否有 core 空闲 bubble

## 12. 常见错误

### 把 pub / sub 当成芯片间通信

`pub / sub` 只描述 chip 内 core 之间的 tensor 版本依赖。跨 chip 数据移动应使用多芯片数据搬运 API，不能用 `pub / sub` 代替。

### 试图共享 core 私有 OCM

私有 OCM 中的 tensor 对其他 core 不可见。跨 core 共享数据应放在 `SHARED_OCM`。

### 多个 core 写同一输出区域

会导致写冲突或非确定结果。为每个 core 划分明确输出 owner。

### pub / sub 作用域不匹配

如果 `pub` 在子图或父作用域中，`sub` 在另一个无法看见该发布的作用域中，构建会报错。应保证发布和订阅处于相同的控制流作用域。

### 假设 `program_id` 自动切分 tensor

Grid 只提供坐标。Kernel 内必须显式用 `program_id` 计算 slice。

## 13. 小结

| 目标 | 做法 |
|---|---|
| 并行处理数据 | Program Grid + slice |
| core 间共享数据 | `SHARED_OCM` |
| 声明写入完成 | `pub` |
| 声明后续读取依赖 | `sub` |
| 避免写冲突 | 明确输出 / shard owner |
| 提升利用率 | 均匀划分 + 合理 tile |
