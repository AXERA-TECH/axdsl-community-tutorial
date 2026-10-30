# 结构化控制流与动态执行

AXDSL 的控制流不是把任意 Python `if` / `while` 自动编译到 NPU，而是提供结构化 API，把 true/false branch 或 loop body trace 成 Graph 子图，再由运行期 tensor 决定执行路径。

## 1. 两类控制流

### 编译期 Python 控制流

条件或循环次数由 Python 常量决定：

```python
TILE_SIZE = 128

if TILE_SIZE > 64:
    ...

for i in range(0, M, TILE_SIZE):
    ...
```

这些逻辑在 trace 时确定，会直接改变生成的 IR。

### 运行期 tensor 控制流

条件来自设备 tensor：

```python
cond = ax.dsl.load(flag)
```

不能写：

```python
if cond:  # 不能读取运行期值
    ...
```

应使用：

```python
ax.dsl.if_op(cond, ...)
```

## 2. `if_op`

### 签名概览

```python
ax.dsl.if_op(
    cond,
    r,
    true_func,
    false_func,
    *operands,
)
```

- `cond`：条件 tensor，必须也出现在 operands 中
- `true_func` / `false_func`：两个都会被 trace 成子图
- `*operands`：传入 branch 的所有读写 tensor
- `r`：可选输出 tensor
- 返回 branch 输出 tensor 列表

### 示例

```python
@ax.builder
def branch_kernel(x, y, flag, output):
    x_ocm = ax.dsl.load(x)
    y_ocm = ax.dsl.load(y)
    cond = ax.dsl.load(flag)

    result = ax.dsl.create_tensor(shape=x_ocm.shape, dtype=x_ocm.dtype)

    def true_func(cond, x_ocm, y_ocm, result):
        ax.dsl.add(x_ocm, y_ocm, r=result)
        return cond, x_ocm, y_ocm, result

    def false_func(cond, x_ocm, y_ocm, result):
        ax.dsl.sub(x_ocm, y_ocm, r=result)
        return cond, x_ocm, y_ocm, result

    ax.dsl.if_op(
        cond,
        result,
        true_func,
        false_func,
        cond,
        x_ocm,
        y_ocm,
        result,
    )

    ax.dsl.store(x=result, r=output)
    return output
```

对应 Python 语义：

```python
if flag:
    result = x + y
else:
    result = x - y
```

### 关键规则

1. `cond` 必须在 operands 中。
2. branch 内需要读写的 tensor 都应作为 operands 传入。
3. 两个 branch 都会被 trace。
4. branch 的输出契约应一致，特别是 `r` 指定的结果。
5. 不存在“只 trace 实际执行分支”的语义。

## 3. `while_loop`

### 签名概览

```python
ax.dsl.while_loop(cond, body_func, *operands)
```

- `cond`：循环条件 tensor，必须包含在 operands 中
- `body_func`：loop body，接收 operands 并按相同顺序返回更新后的 tensors
- 循环在 cond 变为 0 时终止
- 返回更新后的 operands

### 最小示例

```python
@ax.builder
def increment_kernel(x, n, output):
    x_ocm = ax.dsl.load(x)
    cond = ax.dsl.load(n)
    one = ax.dsl.from_numpy(np.array([1], dtype=np.int32))

    def body(cond, x_ocm, one):
        x_ocm += one
        cond -= one
        return cond, x_ocm, one

    _, result, _ = ax.dsl.while_loop(cond, body, cond, x_ocm, one)

    ax.dsl.store(x=result, r=output)
    return output
```

### 状态规则

body 的入参和返回值按位置对应：

```python
def body(a, b, c):
    ...
    return a_new, b_new, c_new
```

如果某个状态不变，也要返回原 tensor。

### 常见错误

- 忘记更新 `cond`
- body 返回数量或顺序不一致
- shape / dtype 改变
- 读写外部 tensor 却没有作为 operand 传入
- cond 永远不为 0，造成死循环

## 4. `range`

### 签名概览

```python
ax.dsl.range(
    index,
    start,
    stop,
    step,
    body_func,
    *operands,
    unroll_factor=1,
)
```

要求：

- `index`、`start`、`stop`、`step` 都是 shape 为 `(1,)` 的 INT32 tensor
- body 第一个参数是 induction variable
- 其余参数是 loop-carried state
- body 返回值数量与 state 数量一致
- body 不能返回 induction variable
- `step` 不能为 0

### 示例骨架

```python
index = ax.dsl.zeros(shape=(1,), dtype=ax.int32)
start = ax.dsl.from_numpy(np.array([0], dtype=np.int32), prefix="start")
stop = ax.dsl.from_numpy(np.array([TILES], dtype=np.int32), prefix="stop")
step = ax.dsl.from_numpy(np.array([1], dtype=np.int32), prefix="step")


def body(index, state, output):
    ...
    return state, output

ax.dsl.range(index, start, stop, step, body, state, output, unroll_factor=1)
```

### 与 Python range 的区别

| Python `for` | AXDSL `range` |
|---|---|
| trace 时可能展开 | 记录为 `AxRange` 子图 |
| 边界通常是编译期值 | 边界可以是运行期 tensor |
| 可返回任意 Python 值 | body 返回 loop-carried tensors |
| 不需要显式 index tensor | 需要 index tensor |
| 不涉及后端展开 | 可通过 `unroll_factor` 请求优化 |

### unroll_factor

`unroll_factor` 用于向图变换请求循环展开 / lane cloning。增大该值可能：

- 提高 ILP 或调度机会
- 增加 OCM 压力
- 增大编译产物
- 改变尾块处理

应基于性能报告选择，而不是盲目最大化。

## 5. `switch`

### 签名概览

```python
ax.dsl.switch(selector, branches, *operands)
```

要求：

- `selector` 是运行期 branch-table index
- `branches` 是 dict
- keys 必须是 dense integer `0..N-1`
- 不支持 default branch
- 每个 branch callable 接收相同 operands
- 每个 branch 必须返回相同且有序的 tensor 列表

### 示例骨架

```python
def case0(x, y, out):
    ax.dsl.add(x, y, r=out)
    return x, y, out

def case1(x, y, out):
    ax.dsl.sub(x, y, r=out)
    return x, y, out

ax.dsl.switch(
    selector,
    {0: case0, 1: case1},
    x,
    y,
    out,
)
```

### 适用场景

- MoE expert 路由
- 多策略 kernel
- runtime layout / mode 选择
- 分支数量固定且 case id 连续

## 6. 动态读写

结构化循环通常配合 `dynamic_ld` / `dynamic_st` 使用。

### dynamic_ld

```python
tile = ax.dsl.dynamic_ld(
    x,
    read_offset,
    tile_shape,
    dim=(0,),
    output_shape=(1, M, N),
)
```

适合：

- KV cache 读取
- dynamic sequence tile
- paged attention
- 动态 batch 内部块

### dynamic_st

```python
ax.dsl.dynamic_st(
    result,
    source_offset,
    write_offset,
    tile_shape,
    dim=(0,),
    output_shape=output.shape,
    r=output,
)
```

适合：

- 写回动态 token 区间
- 循环中累积结果
- paged KV 写入
- 不规则输出位置

## 7. 控制流调试方法

### 1. 先用 PYRUN 验证

控制流 API 的 body 契约容易出错。PYRUN 能先确认 Python 语义。

### 2. 只保留一个循环或分支

把嵌套控制流拆成阶段：

```text
range only
range + dynamic_ld
range + dynamic_ld + dynamic_st
while + if
```

### 3. 检查返回值数量

常见报错：

```text
AxRange body must return one iter_output per iter_arg
```

表示 body 返回值与 loop state 数量不一致。

### 4. 画出状态表

对每个 loop body 列表：

| 输入 state | 更新方式 | 返回值 |
|---|---|---|
| `i` | add one | `i` |
| `acc` | add tile | `acc` |
| `stop` | 不变 | `stop` |

这个表能显著减少顺序错误。

### 5. 用固定 case 验证

先固定 selector / cond / 循环次数：

```python
flag = torch.tensor([1], dtype=torch.int32)
n = torch.tensor([3], dtype=torch.int32)
```

再测试 0、1、非整除和较大值。

## 8. 性能考虑

- 控制流提升表达力，但可能降低静态调度可见性
- `while_loop` 的循环次数在运行期确定，可能限制编译期优化
- `range` 更容易展开和调度
- branch 内 tensor 和 OCM 需要按所有分支可能的工作集评估
- 多芯片 kernel 中控制流与通信 signal 交错时，先用小规模验证死锁
- `unroll_factor` 应通过 cycle / traffic / OCM 综合选择

## 9. 小结

| 需求 | API |
|---|---|
| runtime 二选一 | `if_op` |
| runtime 多分支 | `switch` |
| 运行期次数循环 | `while_loop` |
| counted loop | `range` |
| 动态 offset 读 | `dynamic_ld` |
| 动态 offset 写 | `dynamic_st` |

AXDSL 控制流的核心是：**Python 函数负责描述子图，runtime tensor 负责决定执行哪个子图或循环多少次。**
