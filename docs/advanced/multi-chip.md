# 多芯片与 Neulink 通信

AX8860 的重要能力是多个 chip 协同。相比单 chip Kernel，多芯片编程需要额外回答：

- 每个 chip 持有哪些输入 / 输出？
- 每个 chip 上的 program 处理哪部分？
- 哪些数据需要发送给其他 chip？
- 远端写入何时完成？
- 接收方如何知道数据 ready？

## 1. 拓扑模型

每个 chip 有：

- 本地 DDR
- 本地 core 与 OCM
- 本地 Shared OCM
- Neulink 通信模块

简化图：

```text
Chip 0 DDR ←──Neulink──→ Chip 1 DDR
     │                      │
  4 cores                4 cores
  Shared OCM             Shared OCM
```

当前教程把 chip 视为参与计算的 processing element：

```python
pe = chip_id
```

具体物理拓扑、带宽和路由由硬件与工具链决定。

## 2. 初始化与 launch

```python
CHIP_NUM = 4
ax.init(chip_num=CHIP_NUM, core_num=1)
```

多芯片通常对每个 chip 调用一次同一个 Kernel builder：

```python
for chip_id in range(CHIP_NUM):
    kernel[1](
        x_chip[chip_id],
        output_chip[chip_id],
        signal,
        chip_id,
        CHIP_NUM,
    )
```

关键点：

- `chip_id` 和 `CHIP_NUM` 可以作为 Kernel 参数
- 每个 chip 传入自己的输入 / 输出视图
- AXDSL 依次记录每个 chip 的图
- 所有 chip 记录完成后，再统一执行多芯片仿真
- 当前源码 CRUN 多芯片路径要求每 chip 1 core；能力以目标工具链为准

## 3. Host 侧共享对象

多芯片通信需要 shared tensor 或 signal。Host 侧可创建：

```python
signal = ax.host.shmem_create_signal()
shared = ax.host.shmem_create_tensor(shape=(M, K), dtype=x.dtype)
```

创建后作为 Kernel 参数传入。

::: warning
创建时机和传参方式应与目标工具链版本保持一致。不要在不确定地址归属时自行假设 shared tensor 的物理位置。
:::

## 4. 通信 API

### `ax.parallel.shmem_put`

```python
ax.parallel.shmem_put(dst=remote_dst, src=local_src, pe=remote_chip_id)
```

含义：把本地 `src` 写到 `pe` chip 上的 `dst`。这是阻塞 put 语义，源码文档要求源和目标都是 DDR tensor。

### `ax.parallel.shmem_signal_update`

```python
ax.parallel.shmem_signal_update(
    signal=signal,
    value=1,
    pe=remote_chip_id,
    sig_op=ax.SigOp.SET,
)
```

更新远端 chip 的 signal，常用于 `shmem_put` 后通知接收方。

### `ax.parallel.signal_wait`

```python
ax.parallel.signal_wait(signal=signal, value=1)
```

等待本地 signal 达到指定值。

### `ax.parallel.shmem_put_signal_nbi`

```python
ax.parallel.shmem_put_signal_nbi(
    dst=remote_dst,
    src=local_src,
    pe=remote_chip_id,
    signal=remote_signal,
    value=1,
    sig_op=ax.SigOp.ADD,
)
```

组合数据写入与 signal 更新，常见于通信计算融合场景。

### `pub` / `sub`

源码还提供 pub/sub 风格接口。建议在基础教程中优先讲解 put/signal 模型，高级通信章节再展开 pub/sub。

## 5. 信号配对原则

每一次 remote write 都应有明确的完成通知。

基础模式：

```text
Producer chip
  1. 准备本地数据
  2. shmem_put(dst, src, pe=consumer)
  3. shmem_signal_update(consumer_signal, value, pe=consumer)

Consumer chip
  1. signal_wait(local_signal, value)
  2. 读取 dst
```

不要只依赖调用顺序。多 chip 执行时，代码顺序不等于全局时间顺序。

## 6. Ring All-Gather 模式

假设 4 个 chip，每个 chip 初始持有 `x` 的四分之一。目标：每个 chip 都获得完整 `x`。

```text
初始:
chip0: x0
chip1: x1
chip2: x2
chip3: x3

step0:
chip0 → chip1: x0
chip1 → chip2: x1
chip2 → chip3: x2
chip3 → chip0: x3

step1:
chip0 → chip1: x3
chip1 → chip2: x0
chip2 → chip3: x1
chip3 → chip0: x2

...

step3:
每个 chip 拥有 x0 x1 x2 x3
```

源码示例：

```python
for step in range(CHIP_NUM):
    # 计算本 step 需要转发的数据块
    ...

    axd.shmem_put(
        dst=dst,
        src=src,
        pe=right_chip_id,
    )
    axd.shmem_signal_update(
        signal=signal,
        value=step + 1,
        pe=right_chip_id,
        sig_op=ax.SigOp.SET,
    )
    axd.signal_wait(signal=signal, value=step + 1)
```

完整讲解见 [Ring All-Gather 教程](/tutorials/ring-all-gather)。

## 7. 多芯片 GEMM

典型数据分布：

```text
x: M × K，按 M 维切到各 chip
w: K × N，按 N 维切到各 chip
output: M × N，按 N 维切到各 chip
```

每个 chip 拥有：

```text
sub_x: M/4 × K
sub_w: K × N/4
sub_r: M × N/4
```

要计算完整 `sub_r = x @ w`，需要 all-gather `x` 到每个 chip，再与本 chip 的 `sub_w` 相乘。

进阶版本可以把 all-gather 和 matmul 融合：

```text
每 step:
  等待可用 x block
  计算部分 GEMM
  向邻居转发下一 block
```

这能减少“先完整通信、再计算”的串行时间。

## 8. 输出组织

常见输出布局：

### chip 分片输出

```python
output_ax = torch.empty((CHIP_NUM, M, K))
kernel[1](sub_x, output_ax[chip_id], signal, chip_id, CHIP_NUM)
```

每个 chip 写自己的 `output_ax[chip_id]`。

### 全局输出列分片

```python
output = torch.empty((M, N))
sub_output = output[:, chip_id * n : (chip_id + 1) * n]
```

每个 chip 写全局输出的不同列。

### 每 chip 保存完整副本

all-gather 后每个 chip 可以拥有完整数据，但最终只应有一个明确 owner 写回全局输出，或每个 chip 写自己的副本。

## 9. 死锁与错误排查

### 死锁常见原因

- A 等 B 的 signal，B 等 A 的同一个 signal
- signal value 永远达不到
- 不同 chip 的 step 顺序不一致
- remote signal 更新方向写反
- 多 producer 使用 SET 导致互相覆盖

### 排查步骤

1. 画出每个 step 的通信表
2. 标明每条消息的 producer、consumer、signal、value
3. 确认每个 wait 都有对应 update
4. 从 2 chip / 1 step 开始
5. 固定输入并缩小 shape
6. 打开 trace 查看 chip / job 时间线

### 通信表模板

| Step | Sender | Receiver | Data | Signal | Value |
|---|---|---|---|---|---|
| 0 | chip0 | chip1 | x0 | ready | 1 |
| 0 | chip1 | chip2 | x1 | ready | 1 |
| ... | ... | ... | ... | ... | ... |

## 10. 性能优化

### 通信量

尽量发送必要数据，避免广播完整大 tensor。

### chunk 粒度

太小：signal 和调度开销高。  
太大：难以与计算 overlap，OCM 压力大。

### 通信与计算 overlap

将 all-gather 分块：

```text
收到 block0 → 计算 block0
同时转发下一块
```

### 拓扑利用

ring neighbor 通信通常更容易推理。跨多跳通信是否直接可用、是否更优，应基于硬件文档与 trace 判断。

### 避免不必要的本地 DDR 中转

如果数据已在 shared memory / OCM，可研究是否支持直接作为通信源。具体源和目标 memory type 限制以 API 文档为准。

## 11. 多芯片 checklist

- [ ] `ax.init(chip_num=CHIP_NUM)`
- [ ] 每个 chip 调用一次 kernel
- [ ] chip_id 正确传入
- [ ] 输入 / 输出分片无重叠
- [ ] remote `pe` 计算正确
- [ ] 每次 write 配对 signal
- [ ] signal value 单调或有明确阶段语义
- [ ] 每个 wait 可达
- [ ] 从 2 chip 最小 case 开始
- [ ] reference 按完整数据计算
- [ ] 输出只写全局位置一次或使用支持累加的 API

## 12. 小结

AXDSL 多芯片编程的核心是：

```text
chip-local compute + explicit remote put + explicit signal synchronization
```

它不要求开发者手写底层消息路由，但仍要求清楚定义数据所有权、通信顺序和同步条件。这正是多芯片 Kernel 可靠优化的基础。
