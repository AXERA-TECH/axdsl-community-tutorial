# Ring All-Gather：多芯片通信

All-Gather 是多芯片并行的基础通信模式：开始时每个 chip 只有一部分数据，结束后每个 chip 都拥有完整数据。本教程基于源码示例 `all_gather_one_way.py`，讲解 ring 转发和 signal 同步。

## 1. 目标

假设：

```text
CHIP_NUM = 4
x shape = (M, K)
```

初始每个 chip 拥有 `M / 4` 行：

```text
chip0: x[0:M/4, :]
chip1: x[M/4:2M/4, :]
chip2: x[2M/4:3M/4, :]
chip3: x[3M/4:M, :]
```

目标每个 chip 都拥有完整 `x`。

示例输出：

```text
output shape = (CHIP_NUM, M, K)
output[chip_id] = broadcast x
```

## 2. Ring 拓扑

每个 chip 有右邻居：

```text
right = (chip_id + 1) % CHIP_NUM
```

4 chip 环：

```text
chip0 → chip1 → chip2 → chip3 → chip0
```

## 3. 通信步骤

每个 step：

1. 找到当前 chip 需要发送的数据块
2. `shmem_put` 发给右邻居
3. 更新右邻居的 signal
4. 等待本地 signal 达到当前 step 值
5. 进入下一个 step

以 chip0 为例：

| Step | 发送前持有 | 发给右邻居 chip1 | 转发内容来源 |
|---|---|---|---|
| 0 | x0 | x0 | kernel 输入 |
| 1 | x0, x3 | x3 | output 中已收到部分 |
| 2 | x0, x3, x2 | x2 | output 中已收到部分 |
| 3 | x0, x2, x3, x1 | x1 | output 中已收到部分 |

其他 chip 同理。4 步后，每个 chip 都拥有所有块。

## 4. Kernel 代码骨架

```python
@ax.builder(run_mode=ax.RunMode.CRUN)
def ax_custom_op_kernel(sub_x_chip, output, signal, chip_id, CHIP_NUM):
    m, K = sub_x_chip.shape
    M = m * CHIP_NUM
    right_chip_id = (chip_id + 1) % CHIP_NUM

    for step in range(CHIP_NUM):
        M_start = ((chip_id - step) * M // CHIP_NUM + M) % M

        if step == 0:
            src = sub_x_chip
        else:
            src = output[M_start : M_start + M // CHIP_NUM, :]

        dst = output[M_start : M_start + M // CHIP_NUM, :]

        axd.shmem_put(dst=dst, src=src, pe=right_chip_id)

        axd.shmem_signal_update(
            signal=signal,
            value=step + 1,
            pe=right_chip_id,
            sig_op=ax.SigOp.SET,
        )

        axd.signal_wait(signal=signal, value=step + 1)

    return output
```

::: note
这里的 Python `for step in range(CHIP_NUM)` 是编译期循环，因为 `CHIP_NUM` 是普通 Python 参数。它展开的是通信步骤，不是运行期 tensor 循环。
:::

## 5. Host 侧准备

```python
CHIP_NUM = 4
M = 4
K = 1024

x = torch.rand((M, K), dtype=torch.float32)
output = torch.empty((CHIP_NUM, M, K), dtype=torch.float32)
signal = ax.host.shmem_create_signal()
```

初始化：

```python
ax.init(chip_num=CHIP_NUM, core_num=1)
```

## 6. 每个 chip launch 一次

```python
for chip_id in range(CHIP_NUM):
    sub_x_chip = x[
        chip_id * M // CHIP_NUM :
        (chip_id + 1) * M // CHIP_NUM,
        :,
    ]
    ax_custom_op_kernel[1](
        sub_x_chip,
        output[chip_id, :, :],
        signal,
        chip_id,
        CHIP_NUM,
    )
```

注意：

- 每个 chip 的输入是完整 `x` 的不同行
- 每个 chip 的输出是自己的副本 `output[chip_id]`
- `chip_id` 和 `CHIP_NUM` 是普通 Python 参数，在 Kernel 内可参与逻辑

## 7. 验证

Reference 是同一个 `x` 广播到每个 chip：

```python
output_gt = torch.broadcast_to(x, (CHIP_NUM, M, K))
torch.testing.assert_close(output_gt, output_ax, atol=1e-5, rtol=1e-5)
```

运行本站示例：

```bash
python3 docs/examples/ring_all_gather.py
```

该示例需要 CRUN 与多芯片仿真工具链支持。

## 8. Signal 配对表

以 chip0 发送给 chip1 为例：

| Step | Sender | Receiver | Signal 操作 | Wait |
|---|---|---|---|---|
| 0 | chip0 | chip1 | update right signal to 1 | chip1 waits 1 |
| 1 | chip0 | chip1 | update right signal to 2 | chip1 waits 2 |
| 2 | chip0 | chip1 | update right signal to 3 | chip1 waits 3 |
| 3 | chip0 | chip1 | update right signal to 4 | chip1 waits 4 |

由于 `SigOp.SET` 每次设置为不同阶段值，接收方可以明确判断新数据是否到达。

如果使用 `SigOp.ADD`，signal value 单调递增，也可以表达 ready 数量。不要在一个循环中混用语义不明的 SET / ADD。

## 9. 为什么第一步 src 是输入

Step 0 时，每个 chip 只有本地输入：

```python
if step == 0:
    src = sub_x_chip
```

从 step 1 开始，需要转发的数据已经在本地 output 中：

```python
src = output[M_start:M_start + M // CHIP_NUM, :]
```

这就是 ring all-gather 的核心：每 chip 每步把新收到的块继续传给右邻居。

## 10. 死锁检查

问以下问题：

1. 每个 chip 是否都在每个 step 发送？
2. 每个 wait 的 value 是否能被某个邻居更新？
3. 更新的 `pe` 是否指向正确 chip？
4. `step + 1` 是否保证不同阶段不同 value？
5. 所有 chip 的 `CHIP_NUM` 是否一致？
6. 输出 slice 是否与发送 slice 一致？
7. 是否有 chip 只 wait 不 send？
8. 是否存在两个 producer 对同一 signal 使用 SET 覆盖？

## 11. 扩展：All-Gather + MatMul

得到完整 `x` 后，可以直接做本地 GEMM：

```python
x_full = output  # 当前 chip 的完整副本
sub_r = ax.dsl.matmul(x_full, sub_w)
```

进阶做法是分块 all-gather：

```text
step:
  等待一个 x block ready
  用该 block 计算部分 matmul
  同时把下一 block 发给右邻居
```

这样通信和计算可以 overlap，减少整体等待。

## 12. 练习

### 练习 1：2 chip 版本

把 `CHIP_NUM` 改为 2，画出 step 0 / step 1 的通信表。

### 练习 2：改变方向

把右邻居改为左邻居：

```python
left = (chip_id - 1 + CHIP_NUM) % CHIP_NUM
```

检查 signal 和数据 slice 是否仍正确。

### 练习 3：添加 device print

使用 `device_print` 输出 chip_id / step，确认执行顺序。

### 练习 4：通信计算融合

参考源码：

```text
axnn/axnn/dsl/demo/parallel_demo/all_gather_matmul_fused.py
```

比较“先 all-gather 再 matmul”与“分块 overlap”的 trace 差异。

### 练习 5：trace 分析

打开：

```bash
DUMP_NPU_TRACE=1
DEBUG_SIM_CYCLE_TRACE=1
```

观察通信 job、计算 job 和等待时间。

## 13. 小结

Ring All-Gather 的关键不是记住 `M_start` 公式，而是理解：

```text
本地数据块
  + ring neighbor
  + remote put
  + signal update/wait
  = 多芯片协同数据流
```

掌握该模式后，可以进一步理解 multi-chip GEMM、tensor parallel 和通信计算融合。
