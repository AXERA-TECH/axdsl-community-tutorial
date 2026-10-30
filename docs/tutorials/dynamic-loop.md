# Dynamic Loop：动态循环与动态读写

本教程基于源码示例 `demo_range.py`，讲解如何用：

- `ax.dsl.range`
- `ax.dsl.dynamic_ld`
- `ax.dsl.dynamic_st`

处理 tile 循环中由运行期 index 决定的读写位置。

## 1. 为什么不用普通 Python for

如果循环边界是编译期常量，普通 Python `for` 可以被展开。但动态场景中常见需求是：

- tile 数运行期决定
- KV cache 长度动态
- token 数动态
- paged index 每次不同
- 写回位置来自 index tensor

这时需要结构化控制流和动态读写 API。

## 2. 目标循环

源码示例的 body 每个 iteration：

```text
read one tile a[i]
计算 add / sub / matmul / mul
write result[i]
```

Python reference：

```python
def torch_reference(a):
    output = torch.zeros_like(a)
    for tile_idx in range(START, STOP, STEP):
        a_tile = a[tile_idx]

        add_tmp = a_tile + a_tile
        sub_tmp = add_tmp - a_tile
        matmul_tmp = sub_tmp @ a_tile.transpose(-1, -2)
        output[tile_idx] = matmul_tmp * sub_tmp
    return output
```

## 3. 控制 flow 状态

示例声明：

```python
TOTAL_TILES = 10
START = 0
STOP = TOTAL_TILES
STEP = 1
```

Kernel 内创建 INT32 标量 tensor：

```python
index = ax.dsl.zeros(shape=(1,), dtype=ax.int32)
range_start = ax.dsl.from_numpy(
    np.array([START], dtype=np.int32), prefix="range_start"
)
range_stop = ax.dsl.from_numpy(
    np.array([STOP], dtype=np.int32), prefix="range_stop"
)
range_step = ax.dsl.from_numpy(
    np.array([STEP], dtype=np.int32), prefix="range_step"
)
```

## 4. 动态读写描述

还需要 offset 和 tile shape 描述：

```python
zero_offset = ax.dsl.from_numpy(
    np.zeros(shape=(1,), dtype=np.uint32), prefix="for_zero_offset"
)
tile_shape = ax.dsl.from_numpy(
    np.ones(shape=(1,), dtype=np.uint32), prefix="for_tile_shape"
)
```

读取：

```python
a_tile = ax.dsl.dynamic_ld(
    a,
    read_offset,
    tile_shape,
    dim=(0,),
    output_shape=(1, TILE_M, TILE_N),
).reshape(TILE_M, TILE_N)
```

写入：

```python
ax.dsl.dynamic_st(
    result.reshape(1, TILE_M, TILE_N),
    zero_offset,
    read_offset,
    tile_shape,
    dim=(0,),
    output_shape=output.shape,
    r=output,
)
```

## 5. Loop body

```python
def body_func(range_index, a, output, tile_shape, zero_offset):
    read_offset = ax.dsl.cast(range_index, dtype=ax.uint32)

    a_tile = ax.dsl.dynamic_ld(
        a,
        read_offset,
        tile_shape,
        dim=(0,),
        output_shape=(1, TILE_M, TILE_N),
    ).reshape(TILE_M, TILE_N)

    add_tmp = ax.dsl.add(a_tile, a_tile)
    sub_tmp = ax.dsl.sub(add_tmp, a_tile)
    matmul_tmp = ax.dsl.matmul(sub_tmp, a_tile, transposed=True)
    result = ax.dsl.mul(matmul_tmp, sub_tmp, output_dtype=a_tile.dtype)

    ax.dsl.dynamic_st(
        result.reshape(1, TILE_M, TILE_N),
        zero_offset,
        read_offset,
        tile_shape,
        dim=(0,),
        output_shape=output.shape,
        r=output,
    )

    return a, output, tile_shape, zero_offset
```

注意：

- 第一个参数 `range_index` 是 induction variable
- 后四个是 loop-carried state
- 返回值必须与 state 顺序、数量一致
- 不能返回 `range_index`

## 6. 启动 range

```python
ax.dsl.range(
    index,
    range_start,
    range_stop,
    range_step,
    body_func,
    a,
    output,
    tile_shape,
    zero_offset,
    unroll_factor=UNROLL_FACTOR,
)
```

## 7. 完整运行

本站附带示例：

```bash
python3 docs/examples/dynamic_range.py
```

源仓库原始示例：

```bash
python3 axnn/axnn/dsl/demo/control_flow_demo/demo_range.py
```

## 8. 状态表

理解 loop body 契约时，建议画状态表：

| 输入 | 类型 | 是否更新 | 返回 |
|---|---|---|---|
| `range_index` | induction variable | 由 range 管理 | 不返回 |
| `a` | 输入 tensor | 否 | `a` |
| `output` | 输出 tensor | dynamic_st 更新 | `output` |
| `tile_shape` | 描述 tensor | 否 | `tile_shape` |
| `zero_offset` | 描述 tensor | 否 | `zero_offset` |

## 9. PyRun 与 Crun 差异

源码注释中特别提醒：

- PyRun 按 Python range 执行 body，返回值作为下一轮状态
- Crun 在 iteration 末尾可能将结果写回输入槽
- 原地或共享 view 写入可能影响调用方原 tensor
- 如需保留初值，应先 `move(initial, r=state)` 再传入 state

因此不要依赖“调用方原 tensor 一定保持初始值”这个副作用。

## 10. 常见错误

### body 返回数量错误

```text
AxRange body must return one iter_output per iter_arg
```

检查 state 表。

### 重复传入同一个 tensor operand

`AxRange operands must not repeat the same Tensor`。不要把同一个 tensor 以两个状态名传入。

### index / bounds dtype 不对

`range` 要求：

```text
shape = (1,)
dtype = INT32
```

### 返回 induction variable

`range` 自己管理 induction variable，body 不应返回它。

### 动态 offset dtype 不匹配

`dynamic_ld/st` 通常需要 uint32 offset。示例中先：

```python
read_offset = ax.dsl.cast(range_index, dtype=ax.uint32)
```

## 11. 练习

### 练习 1：修改循环次数

把 `TOTAL_TILES` 改为 3、16、31，观察 reference 是否仍通过。

### 练习 2：修改 unroll_factor

比较：

```text
1, 2, 4
```

的 correctness、编译结果和性能。

### 练习 3：替换 body

把 body 改成：

```text
output[i] = a[i] * 2
```

保留 dynamic_ld / dynamic_st 结构。

### 练习 4：运行期 stop

研究如何把 `STOP` 从输入 tensor 转为 shape `(1,)` 的 INT32 DSL tensor，实现运行期循环次数。

## 12. 小结

动态循环的组合套路是：

```text
INT32 index/bounds
  + range 子图
  + uint32 runtime offset
  + dynamic_ld
  + 计算
  + dynamic_st
```

这套结构可以扩展到 KV cache、paged attention、MoE 和变长 token 场景。
