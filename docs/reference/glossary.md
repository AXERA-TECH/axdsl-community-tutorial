# 术语表

| 术语 | 含义 |
|---|---|
| AXDSL | AX8860 NPU 工具链中的 Python 自定义 Kernel DSL |
| AXNN | 提供 AXDSL、backend、编译与运行相关能力的 Python 包 |
| Kernel | 被 `@ax.builder` 包装并可通过 grid 启动的设备逻辑函数 |
| Builder | `@ax.builder` 返回的对象，负责记录、编译、执行和调试 |
| Trace / Record | 执行 Python Kernel 并记录 AXDSL 算子成 Graph IR 的过程 |
| Graph IR | AXDSL 内部计算图表示，由算子、tensor、子图和依赖组成 |
| Grid | 逻辑 program 空间，如一维 `8` 或二维 `(4, 4)` |
| Program | grid 中的一个逻辑任务，通过 `program_id` 获取坐标 |
| Program ID | 当前 program 在指定 grid axis 上的坐标 |
| Num Programs | 指定 grid axis 上的 program 数量 |
| Tile | 从大 tensor 切出的局部工作块 |
| Tiling | 将输入 / 输出划分为 tile 的过程 |
| Tail tile | 最后不足一个标准 tile 的块 |
| Caller-allocated output | 输出 buffer 由调用方创建并传入 Kernel |
| Runtime tensor | 元数据在编译期可见、元素值运行期才可用的 tensor |
| Compile-time constant | trace 时已确定的 Python / NumPy 常量 |
| `const_inputs` | builder 参数，用于指定按常量处理的输入 |
| DDR | chip 外部内存，常用作输入输出和大 buffer |
| OCM | core 私有片上内存 |
| Shared OCM | chip 内多 core 共享片上内存 |
| Signal | 用于同步状态的地址空间 / 对象 |
| Load | DDR 数据载入局部计算内存 |
| Store | 局部结果写回 DDR |
| Move | 数据移动或写入指定目标 |
| Dynamic load | 按运行期 offset / shape 读取 |
| Dynamic store | 按运行期 offset / shape 写入 |
| PYRUN | 使用 Python reference implementation 执行 |
| CRUN | 后端编译并使用 simulator 执行 |
| BRUN | 编译 AXModel 后在物理设备 / AXCL runtime 执行 |
| Reference implementation | 用 PyTorch 等实现的正确性基准 |
| Per-layer check | 比较 PYRUN 与 CRUN 的中间层结果 |
| Fuse | 将多个算子合成更少调度单元 |
| Graph transformation | 图级优化或模式重写 |
| Job | 后端调度中的一个执行任务 |
| Traffic | 数据搬运量，常指 DDR 读写 |
| Overlap | 搬运与计算或其他任务在时间上重叠 |
| Bubble | 流水线或多个执行单元中的空闲时间 |
| Engine utilization | 计算引擎利用程度 |
| EU duty | execution unit 占用情况 |
| Cache key | 判断编译缓存是否命中的组合键 |
| AXModel | 可由 runtime 加载执行的编译产物 |
| Neulink | AX8860 chip 间通信模块 |
| PE | Processing Element；多芯片通信中常指目标 chip |
| All-Gather | 使每个参与者最终都拥有完整数据 |
| Ring all-gather | 沿环形拓扑逐 step 转发实现的 all-gather |
| Producer | 数据或 signal 的产生方 |
| Consumer | 等待并使用数据或 signal 的一方 |
| Deadlock | 多个执行方相互等待，无法继续 |
| Owner | 某输出区域唯一或明确负责写入的 program / core / chip |
| Dtype | tensor 数据类型 |
| BF16 | 16-bit brain floating point |
| FP16 | 16-bit IEEE floating point |
| FP8 | 8-bit floating point |
| MX / NVFP | block / microscaling 类低精度格式 |
