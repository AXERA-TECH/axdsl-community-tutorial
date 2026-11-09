# API Cookbook

本章按任务给出最小代码片段。完整签名见[完整算子列表](/reference/operators)。

## 1. 定义并启动 Kernel

```python
import axnn.dsl as ax

@ax.builder(run_mode=ax.RunMode.PYRUN)
def kernel(x, output, tile_size):
    start = ax.program_id(0) * tile_size
    stop = min(start + tile_size, x.shape[0])
    t = ax.dsl.load(x[start:stop])
    y = ax.dsl.mul(t, t)
    ax.dsl.store(x=y, r=output[start:stop])
    return output


ax.init(chip_num=1, core_num=1)
kernel[8](x, output, 128)
```

## 2. Kernel launch 语法

```python
kernel[4](a, output)          # 1D grid，默认 4 core
kernel[4, 1](a, output)       # 1D grid，1 core
kernel[(2, 2),](a, output)    # 2D grid，默认 4 core
kernel[(2, 2), 1](a, output)  # 2D grid，1 core
```

注意：`kernel[2, 2]` 表示一维 grid 大小为 2、core 数为 2，不是二维 grid `(2, 2)`。

## 3. 一维 tile 与尾块

```python
start = ax.program_id(0) * TILE
stop = min(start + TILE, N)
```

## 4. 二维 tile

```python
pid_m = ax.program_id(0)
pid_n = ax.program_id(1)

m0 = pid_m * TILE_M
m1 = m0 + TILE_M
n0 = pid_n * TILE_N
n1 = n0 + TILE_N
```

## 5. DDR → OCM → DDR

```python
x_tile = ax.dsl.load(x[start:stop])
y = ax.dsl.add(x_tile, x_tile)
ax.dsl.store(x=y, r=output[start:stop])
```

## 6. 写入已有 tensor

```python
temp = ax.dsl.create_tensor(shape=x.shape, dtype=x.dtype)
ax.dsl.add(x, y, r=temp)
```

## 7. Elementwise

```python
a = ax.dsl.add(x, y)
s = ax.dsl.sub(x, y)
m = ax.dsl.mul(x, y)
d = ax.dsl.div(x, y)
```

## 8. Activation

```python
y_sigmoid = ax.dsl.sigmoid(x)
y_tanh = ax.dsl.tanh(x)
y_gelu = ax.dsl.gelu(x)
y_silu = ax.dsl.silu(x)
```

## 9. 数值函数

```python
y = ax.dsl.exp(x)
y = ax.dsl.log2(x)
y = ax.dsl.sqrt(x)
y = ax.dsl.reciprocal(x)
y = ax.dsl.floor(x)
y = ax.dsl.ceil(x)
```

## 10. Reduction

```python
s = ax.dsl.sum(x, dim=-1, keepdim=True)
m = ax.dsl.max(x, dim=-1, keepdim=True)
i = ax.dsl.argmax(x, dim=-1)
p = ax.dsl.softmax(x, dim=-1)
```

## 11. 稳定 softmax

```python
m = ax.dsl.max(x, dim=-1, keepdim=True)
shifted = ax.dsl.sub(x, m)
e = ax.dsl.exp(shifted)
s = ax.dsl.sum(e, dim=-1, keepdim=True)
y = ax.dsl.div(e, s)
```

## 12. Matmul

```python
# x: (M, K), w: (N, K)
y = ax.dsl.matmul(x, w, transposed=True)
```

## 13. Tiled matmul

```python
a_tile = ax.dsl.load(a[m0:m1, :])
b_tile = ax.dsl.load(b[n0:n1, :])
c_tile = ax.dsl.matmul(a_tile, b_tile, transposed=True)
ax.dsl.store(x=c_tile, r=output[m0:m1, n0:n1])
```

## 14. Shape

```python
y = ax.dsl.reshape(x, shape=(M, N))
y = ax.dsl.transpose(x, perm=(1, 0))
y = ax.dsl.concat((a, b), dim=0)
y = ax.dsl.pad(x, padding=(0, 1, 0, 0))
```

## 15. Index

```python
y = ax.dsl.gather(x, indices, dim=0)
y = ax.dsl.index_select(x, indices, dim=0)
ax.dsl.scatter(r=output, indices=indices, x=value, dim=0)
ax.dsl.index_add(x=value, indices=indices, r=output, dim=0)
```

## 16. 条件选择

```python
zero = ax.dsl.zeros(shape=x.shape, dtype=x.dtype)
relu = ax.dsl.maximum(x, zero)
clipped = ax.dsl.clip(x, min=min_value, max=max_value)
picked = ax.dsl.where(cond, a, b)
```

## 17. dtype 转换

```python
y = ax.dsl.cast(x, dtype=ax.bf16)
s = ax.dsl.sum(x, dim=-1, output_dtype=ax.fp32)
```

## 18. 常量

```python
import numpy as np

zero = ax.dsl.zeros(shape=(1,), dtype=ax.int32)
one = ax.dsl.full(shape=(1,), fill_value=1, dtype=ax.int32)
step = ax.dsl.from_numpy(np.array([1], dtype=np.int32))
```

## 19. 运行期 if

```python
def true_func(cond, x, y, out):
    ax.dsl.add(x, y, r=out)
    return cond, x, y, out

def false_func(cond, x, y, out):
    ax.dsl.sub(x, y, r=out)
    return cond, x, y, out

ax.dsl.if_op(cond, out, true_func, false_func, cond, x, y, out)
```

## 20. while 循环

```python
def body(cond, state, one):
    state += one
    cond -= one
    return cond, state, one

_, state, _ = ax.dsl.while_loop(cond, body, cond, state, one)
```

## 21. range 循环骨架

```python
def body(index, state, output):
    ...
    return state, output

ax.dsl.range(
    index,
    start,
    stop,
    step,
    body,
    state,
    output,
    unroll_factor=1,
)
```

## 22. 动态读取

```python
tile = ax.dsl.dynamic_ld(
    x,
    read_offset,
    tile_shape,
    dim=(0,),
    output_shape=(1, M, N),
)
```

## 23. 动态写入

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

## 24. 选择 DEVICE_CPU 后端

```python
from axnn.backend import BACKEND

@ax.builder(
    backend=BACKEND.DEVICE_CPU,
    run_mode=ax.RunMode.BRUN,
)
def kernel(...):
    ...
```

## 25. CRUN 正确性检查

```python
@ax.builder(
    run_mode=ax.RunMode.CRUN,
    enable_cache=False,
    check=ax.Check.ALL_CLOSE,
    check_mode=ax.CheckMode.CHECK_PER_LAYER,
    check_rtol=1e-3,
    check_atol=1e-3,
)
def kernel(...):
    ...
```

## 26. 编译缓存

```python
@ax.builder(
    run_mode=ax.RunMode.CRUN,
    enable_cache=True,
    cache_path=".axdsl_cache",
)
def kernel(...):
    ...
```

## 27. 离线编译 AXModel

```python
ax.init(chip_num=1)
compiled_path = kernel[(2, 2), 1].compile(a, b, output)
print(compiled_path)
```

产物路径格式：

```text
<cache_path>/<kernel_name>.<hash>.compiled.axmodel
```

复制到开发板：

```bash
ssh user@board 'mkdir -p ~/.cache/axdsl'
scp ~/.cache/axdsl/kernel.*.compiled.axmodel user@board:~/.cache/axdsl/
```

## 28. 性能导出

```python
@ax.builder(
    run_mode=ax.RunMode.CRUN,
    enable_cache=False,
    dump_dsl_perf=True,
    dump_path="output/kernel",
)
def kernel(...):
    ...
```

## 29. Trace 环境变量

```bash
DUMP_NPU_TRACE=1 DEBUG_SIM_CYCLE_TRACE=1 python3 kernel.py
```

## 30. device_print

```python
ax.dsl.device_print(...)
```

## 31. BRUN session

```python
ax.init(chip_num=1)
ax.init_session()
kernel[grid](...)
ax.synchronize_stream()
ax.finalize()
```
