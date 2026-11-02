# 多芯片编程与 Neulink 数据搬运

AX8860 支持多个 chip 协同。相比单 chip Kernel，多芯片编程需要额外回答：

- 每个 chip 持有哪些输入 / 输出？
- 哪些数据需要发送到其他 chip？
- 远端目标 tensor 在哪个 chip 上？
- 输出全局布局如何避免写冲突？

::: important
`pub / sub` 是 **chip 内核间同步接口**，不是芯片间同步接口。多芯片之间的显式数据搬运使用 `shmem_put` 等 chip 级 API。
:::

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

教程把 chip 视为参与计算的 processing element：

```python
pe = chip_id
```

具体物理拓扑、带宽和路由以硬件与工具链文档为准。

## 2. 初始化与 launch

```python
CHIP_NUM = 2
ax.init(chip_num=CHIP_NUM, core_num=1)
```

多芯片通常对每个 chip 调用一次同一个 Kernel builder：

```python
for chip_id in range(CHIP_NUM):
    kernel[1, 1](
        x_chip[chip_id],
        output_chip[chip_id],
        chip_id,
        CHIP_NUM,
    )
```

关键点：

- `chip_id` 和 `CHIP_NUM` 可以作为 Kernel 参数
- 每个 chip 传入自己的输入 / 输出视图
- AXDSL 依次记录每个 chip 的图
- 当前 CRUN 多芯片路径要求每 chip 1 core
- 具体多 chip 执行能力以目标工具链版本为准

## 3. Host 侧 shared tensor

Host 侧可以创建跨 chip shared DDR tensor：

```python
shared = ax.host.shmem_create_tensor(
    shape=(M, K),
    dtype=x.dtype,
)
```

创建后作为 Kernel 参数传入。

::: warning
创建时机、地址归属和传参方式应与目标工具链版本保持一致。不要在不确定远端地址布局时自行假设 shared tensor 的物理位置。
:::

## 4. `ax.parallel.shmem_put`

`shmem_put` 是 chip 级数据搬运 API：

```python
ax.parallel.shmem_put(
    dst=remote_dst,
    src=local_src,
    pe=remote_chip_id,
)
```

含义：

- `src`：当前 chip 上的源 tensor
- `dst`：目标 chip 上的目标 tensor
- `pe`：目标 chip id
- 源码文档要求 `src` 和 `dst` 均为 DDR tensor
- 该 API 描述一次 blocking put

它用于跨 chip DDR 数据搬运，不是 core 间 `pub / sub` 同步。

## 5. 基本调用形态

`shmem_put` 的调用形态如下：

```python
from axnn.dsl import parallel as axd

axd.shmem_put(
    dst=remote_dst,
    src=local_src,
    pe=remote_chip_id,
)
```

其中 `local_src` 与 `remote_dst` 必须是已经由工具链正确映射的 DDR tensor。实际 Kernel 中还需要结合当前版本的多芯片运行约定，明确远端 buffer 创建、通信顺序和输出 owner；不要把 `pub / sub` 当作跨 chip 完成通知使用。

## 6. 输出组织

常见输出布局：

### chip 分片输出

```python
output = torch.empty((CHIP_NUM, M, K))
kernel[1, 1](sub_x, output[chip_id], chip_id, CHIP_NUM)
```

每个 chip 写自己的 `output[chip_id]`。

### 全局输出列分片

```python
output = torch.empty((M, N))
sub_output = output[:, chip_id * n : (chip_id + 1) * n]
```

每个 chip 写全局输出的不同列。

### 每 chip 保存完整副本

All-gather 类算法可能让每个 chip 拥有完整中间数据，但最终全局输出仍应有明确 owner，或每个 chip 写自己的独立副本。

## 7. 多芯片 GEMM 的数据分布

典型分布：

```text
x: M × K，按 M 维切到各 chip
w: K × N，按 N 维切到各 chip
output: M × N，按 N 维切到各 chip
```

每个 chip 拥有：

```text
sub_x: M / CHIP_NUM × K
sub_w: K × N / CHIP_NUM
sub_r: M × N / CHIP_NUM
```

要计算完整 `sub_r = x @ w`，需要让每个 chip 获得需要的 `x` 分块，再与本 chip 的 `sub_w` 相乘。

这个章节只描述数据分布。完整的通信调度、完成通知和 overlap 策略应使用当前工具链正式支持的多芯片编程接口。

## 8. pub / sub 与 shmem_put 的区别

| 项目 | `pub / sub` | `shmem_put` |
|---|---|---|
| 作用范围 | 单 chip 内 core 之间 | chip 之间 |
| 作用对象 | DSL tensor 版本事件 | DDR 数据块 |
| 表达内容 | 写入发布与后续读取订阅 | 远端数据搬运 |
| 典型内存 | `SHARED_OCM` / 非 OCM tensor | DDR |
| 是否用于芯片间同步 | 否 | 用于跨 chip 数据搬运 |

错误示例：

```python
# 错误理解：pub/sub 不能把数据同步到另一个 chip。
axd.pub(shared_tensor)
# 另一个 chip 上 axd.sub(shared_tensor) 并不会获得本 chip 数据。
```

正确分类：

```text
chip 内 core 协作：pub / sub
chip 间数据搬运：shmem_put
```

## 9. 排查多芯片问题

### 数据 ownership

为每个输出区域列表：

| 输出区域 | Owner chip | 写入方式 |
|---|---|---|
| `output[:, 0:N/2]` | chip 0 | local compute |
| `output[:, N/2:N]` | chip 1 | local compute |

### 通信表

| Step | Source chip | Destination chip | Source slice | Destination slice |
|---|---|---|---|---|
| 0 | chip0 | chip1 | `x[0:M/2]` | `remote_x[0:M/2]` |
| ... | ... | ... | ... | ... |

### 最小复现

从最小配置开始：

```text
1 chip
2 chip
每个 chip 1 core
每个 chip 1 个小块
单一 shmem_put
```

不要一开始就实现复杂多芯片通信或通信计算重叠。

## 10. 性能考虑

- 只发送必要数据，避免广播完整大 tensor
- 控制 chunk 大小，避免任务过碎
- 评估 Neulink 带宽与计算时间
- 查看 trace 中的通信 job 与计算 overlap
- 避免 chip 间访问成为瓶颈
- 减少不必要的本地 DDR 中转

## 11. 多芯片 checklist

- [ ] `ax.init(chip_num=CHIP_NUM)`
- [ ] 每个 chip 调用一次 kernel
- [ ] chip_id 正确传入
- [ ] 输入 / 输出分片无重叠
- [ ] remote `pe` 计算正确
- [ ] `shmem_put` 的 src / dst 均为 DDR tensor
- [ ] 输出只写全局位置一次或写每 chip 独立副本
- [ ] 不把 `pub / sub` 当成芯片间通信
- [ ] 从 2 chip 最小 case 开始
- [ ] reference 按完整数据计算

## 12. 小结

多芯片编程的核心是：

```text
chip-local compute + explicit remote DDR data movement
```

`pub / sub` 只用于 chip 内 core 协作；跨 chip 数据搬运使用 `shmem_put` 等 chip 级 API。完整的通信调度和同步策略应以后端正式支持的接口为准。
