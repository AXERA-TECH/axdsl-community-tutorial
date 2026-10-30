# FAQ 与故障排查

## 安装与环境

### AXDSL 是独立包吗？

AXDSL 包含在 AXNN Python 包中，源码路径通常为：

```text
axnn/axnn/dsl
```

安装方式应与 AXNN 工具链一起提供。

### 为什么 `import axnn.dsl` 失败？

检查：

- Python 环境是否正确
- 是否在仓库根目录执行
- 是否 source 了开发环境脚本
- AXNN 依赖是否完整
- Python 版本是否受支持

### 为什么示例默认用 PYRUN？

PYRUN 不依赖后端编译，适合快速验证 DSL 语义。后端能力仍需 CRUN 验证。

## Kernel 定义

### `@ax.builder` 做了什么？

把 Python 函数包装为 launchable kernel。在 `kernel[grid](...)` 时记录 Graph IR，并根据 run mode 执行。

### 可以在 Kernel 内调用任意 Python 库吗？

不建议。AXDSL 只能保证官方文档允许的语法和 API 被正确记录。第三方库中的运行期 tensor 操作通常不能直接映射。

### 为什么必须 return output？

返回值帮助 builder 确定 Kernel 输出契约，包括输出数量、顺序和元数据。

### Kernel 可以有多个输出吗？

可以，返回 tuple：

```python
return out_a, out_b
```

## 参数

### Torch tensor 和 int 参数有什么区别？

Torch tensor 是 runtime tensor；int / tuple / list 等普通 Python 值是 compile-time constant。

### 为什么 tile size 变化会变慢或重新编译？

tile size 是编译期参数，会影响 IR、任务划分和编译产物，也可能影响 cache key。

### `const_inputs` 什么时候用？

用于权重、lookup table、常量输入等希望按编译期数据处理的参数。

## Grid 与切片

### grid 会自动切 tensor 吗？

不会。grid 只提供逻辑 program 坐标，Kernel 作者需要用 `program_id` 计算 slice。

### `x[i]` 为什么不支持？

整数索引会降低 rank，当前 AXDSL 切片规则要求 rank-preserving。改成 `x[i:i+1]`。

### 怎么处理尾部不完整 tile？

```python
stop = min(start + TILE, x.shape[0])
```

### 为什么输出有未定义值？

可能某些输出区域没有写满。检查 grid 覆盖、尾块和每个 program 的 owner。

## 内存

### load / store / move 分别做什么？

- `load`：DDR → OCM / 计算局部内存
- `store`：OCM 结果 → DDR
- `move`：移动或写入指定目标

### OCM 和 Shared OCM 区别？

OCM 是 core 私有；Shared OCM 是 chip 内多 core 共享。

### 为什么 OCM 超限？

tile、权重、输出和中间 tensor 都会占用工作集。减小 tile 或减少临时变量。

### 中间结果会自动留在 OCM 吗？

DSL 记录算子依赖，后端会调度。显式 `store` 到 DDR 会明确产生外部写回。是否融合取决于图优化与后端支持。

## 控制流

### 普通 `if` 可以用吗？

如果条件是编译期 Python 常量，可以。如果条件是运行期 tensor，用 `ax.dsl.if_op`。

### 普通 `for` 可以用吗？

编译期常量循环可以 trace 展开。运行期循环用 `range` 或 `while_loop`。

### range 的 start / stop / step 有什么要求？

要求 shape `(1,)`、INT32 tensor，且 step 非 0。

### while_loop body 怎么写？

body 接收 loop operands，按相同顺序返回更新后的 tensors，cond 必须包含在 operands 中并由 body 更新。

### range body 报返回值数量错误怎么办？

检查 body 返回值是否与除 induction variable 之外的每个 state 一一对应。

## 执行与验证

### PYRUN 正确但 CRUN 失败，说明什么？

可能是：

- 后端不支持 dtype / shape
- tile / OCM 超限
- 控制流契约问题
- 并行同步错误
- 精度容差过严

开启 `CHECK_PER_LAYER` 定位。

### check 与 cache 为什么不能同时开启？

缓存不会保留 correctness check 所需的全部 debug 信息，因此当前实现禁止同时启用。

### bf16 tolerance 怎么选？

与数据分布、reduce 长度和 reference 有关。教程示例常用 `atol=1e-2` 到 `2e-2`，生产需按任务验证。

## 多核

### 多个 program 会自动使用多个 core 吗？

launch config 提供 grid 和 core 数，后端负责调度。实际可用 core 数受硬件和实现限制。

### core 之间如何共享数据？

使用 Shared OCM，并通过 signal / `sync_all` 建立同步关系。

### 多核偶发错误怎么排查？

检查：

- 输出写冲突
- signal 缺失
- producer/consumer 顺序
- 每个 core 的 slice
- 尾块 owner

## 多芯片

### 多芯片为什么每个 chip 调一次 kernel？

AXDSL 需要为每个 chip 记录对应的 graph 和输入输出视图。

### `pe` 是什么？

通信目标 processing element，在当前教程中通常是目标 chip id。

### signal_wait 卡住怎么办？

检查：

- 是否有远端 update
- `pe` 是否正确
- value 是否可达
- SET / ADD 语义是否一致
- 是否形成循环等待

## 性能

### 增大 tile 一定更快吗？

不一定。tile 影响 OCM、复用、并行度、尾块和 job 形态，应基于 trace 实验。

### 如何判断是 memory bound？

查看 DDR traffic、engine idle、overlap。如果搬运时间长且计算单元空闲，常见为 memory bound。

### 自动调参可用吗？

当前开源源码公开 `ax.builder` 签名没有 `auto_tune_cases`。不要在社区首版文档承诺该 API。可在脚本层枚举参数并记录结果。

### 缓存没有命中怎么办？

检查：

- kernel source / AST 是否变化
- shape / dtype / grid / 常量是否变化
- 工具链版本是否变化
- cache path 是否正确
- 是否使用不同编译配置

## 报错速查

| 现象 | 优先检查 |
|---|---|
| slice / shape 错 | program_id 映射、尾块、rank-preserving slice |
| 输出全 0 或随机 | 输出是否写满、是否使用 empty |
| OCM 超限 | tile、临时 tensor、权重 |
| check 失败 | dtype、tolerance、reference、逐层结果 |
| graph 记录失败 | 普通 Python 控制流、第三方库调用 |
| range 契约失败 | state 数量、返回顺序、INT32 标量 |
| while 死循环 | cond 是否更新且最终为 0 |
| 多核旧值 | signal wait / update 配对 |
| 多芯片卡死 | 通信表、pe、signal value |
