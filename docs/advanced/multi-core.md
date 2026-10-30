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

注意：`program_id` 是 trace 时的逻辑坐标，Kernel 代码应根据它切分数据。不同 program 会记录 / 生成对应的数据访问关系。

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
kernel[grid, core_num](...)
```

例如：

```python
kernel[(8, 8), 4](...)
```

如果省略 `core_num`，默认为 grid 元素数量。实际调度受硬件 core 数与后端实现限制。

## 4. 独立并行模式

如果 core 之间没有依赖，这是最理想的形式：

```text
core 0: input slice 0 → compute → output slice 0
core 1: input slice 1 → compute → output slice 1
core 2: input slice 2 → compute → output slice 2
core 3: input slice 3 → compute → output slice 3
```

特点：

- 无通信
- 无 signal
- 无写冲突
- 负载由数据划分决定

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

## 5. Shared OCM 协作

core 私有 OCM 不能被其他 core 直接访问。要交换数据，需要 Shared OCM。

典型流程：

```text
Producer core
  计算 / load
  move 到 Shared OCM
  更新 signal

Consumer core
  等待 signal
  从 Shared OCM move 到私有 OCM
  计算
```

### 创建 Shared OCM tensor

```python
from axnn.backend.base import MemoryType

shared = ax.dsl.create_tensor(
    shape=x.shape,
    dtype=x.dtype,
    memory_type=MemoryType.SHARED_OCM,
)
```

### 同步 API

常用：

```python
ax.parallel.shmem_create_signal()
ax.parallel.signal_update(signal, value)
ax.parallel.signal_wait(signal, value)
ax.parallel.sync_all()
```

`sync_all` 同步当前 chip 内参与 Kernel 的 core。signal 用于表达更细粒度的数据 readiness。

## 6. 最小 producer-consumer 示例

以下示例来自源码测试中的 local signal 模式：core 0 把输入写入 Shared OCM，其他 core 等待后读取。

```python
from axnn.backend.base import MemoryType
import axnn.dsl as ax
from axnn.dsl import parallel as axd


@ax.builder
def local_signal_kernel(x, out):
    core_id = ax.program_id(0)

    signal = axd.shmem_create_signal()
    shared = ax.dsl.create_tensor(x.shape, x.dtype, MemoryType.SHARED_OCM)

    if core_id == 0:
        t = ax.dsl.load(x)
        ax.dsl.move(t, r=shared)
        axd.signal_update(signal, 1)
    else:
        axd.signal_wait(signal, 1)
        t = ax.dsl.create_tensor(x.shape, x.dtype)
        ax.dsl.move(shared, r=t)
        ax.dsl.store(x=t, r=out)

    return out
```

启动：

```python
ax.init(chip_num=1, core_num=2)
x = torch.randn((2, 16), dtype=torch.float32)
out = torch.empty_like(x)
local_signal_kernel[2](x, out)
```

## 7. Signal 语义

### 更新

```python
axd.signal_update(signal, 1)
axd.signal_update(signal, value, sig_op=ax.SigOp.SET)
```

### 等待

```python
axd.signal_wait(signal, value)
```

### 常见策略

| 策略 | 适合场景 |
|---|---|
| SET 到固定阶段号 | 阶段明确的 pipeline |
| ADD 累加计数 | 多 producer / 多块数据 ready |
| 每块数据独立 signal | 精细同步，但开销更大 |
| 一个 signal 表示批次完成 | 简单，但并行度可能降低 |

选择时要避免：

- 等待永远达不到的 value
- 不同阶段重复使用相同 value
- ADD 后 SET 导致状态回退
- 多 producer 写同一 signal 时未定义顺序

## 8. 并行模式

### Map

每个 core 独立处理相同结构的数据块。

```text
data blocks → core 0..N-1 → output blocks
```

### Reduce

每个 core 计算局部结果，再归约。

```text
core partials → Shared OCM → one core reduce
```

### Pipeline

不同 core 处理不同阶段。

```text
core 0: stage 0
core 1: stage 1
core 2: stage 2
```

需要阶段 signal。

### Producer-consumer

```text
producer core → shared buffer → consumer core
```

### Split-compute-merge

```text
input split
  ↓
cores compute
  ↓
shared partials
  ↓
merge core
```

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
| `acc[indices]` | 多 core | `index_add` 等支持累加语义的 API |

如果 API 没有明确并发写语义，不要假设硬件会自动串行化。

## 11. 多核性能检查清单

- [ ] grid 与数据块一一对应
- [ ] 每个 core 工作量相近
- [ ] 尾块有明确 owner
- [ ] 输出没有写冲突
- [ ] 中间结果是否需要 Shared OCM
- [ ] signal 粒度是否过细
- [ ] 是否存在不必要的 `sync_all`
- [ ] 每个 core OCM 是否超限
- [ ] DDR traffic 是否随 core 数放大
- [ ] 性能报告中是否有 core 空闲 bubble

## 12. 常见错误

### 忘记同步

表现：读取 Shared OCM 得到旧值。

处理：producer 更新 signal，consumer 等待。

### 同步过多

表现：并行度下降，trace 中大量等待。

处理：合并数据块，使用阶段 signal，减少全局 barrier。

### 只在一个 core 上做大量工作

表现：一个 core 忙，其他 core 空闲。

处理：重新划分数据，或将 merge 拆成两级。

### OCM 超限

表现：多 core 编译失败或调度失败。

处理：减少每个 core tile 和临时 tensor。

### 假设 `program_id` 自动切分 tensor

表现：所有 core 写同一输出或只写第一个 tile。

处理：在 Kernel 内显式用 `program_id` 计算 slice。

## 13. 小结

| 目标 | 做法 |
|---|---|
| 并行处理数据 | Program Grid + slice |
| core 间共享数据 | Shared OCM |
| 表示数据 ready | signal |
| 全体 core 同步 | `sync_all` |
| 避免写冲突 | 明确输出 owner |
| 提升利用率 | 均匀划分 + 合理 tile |
