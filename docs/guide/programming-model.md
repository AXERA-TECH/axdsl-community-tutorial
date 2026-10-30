# 编程模型

本章介绍 AXDSL Kernel 的基础规则：如何定义、启动、传参、切片、写输出，以及 Python 语法支持边界。

## 1. 基本模板

```python
import torch
import axnn.dsl as ax


@ax.builder(run_mode=ax.RunMode.PYRUN)
def my_kernel(x, output, tile_size):
    start = ax.program_id(0) * tile_size
    stop = min(start + tile_size, x.shape[0])

    x_tile = ax.dsl.load(x[start:stop])
    result = ax.dsl.mul(x_tile, x_tile)
    ax.dsl.store(x=result, r=output[start:stop])
    return output


def run(x, output):
    tile_size = 128
    grid = (x.shape[0] + tile_size - 1) // tile_size
    my_kernel[grid](x, output, tile_size)
```

## 2. 初始化

```python
ax.init(chip_num=1, core_num=1)
```

| 参数 | 说明 |
|---|---|
| `chip_num` | 参与本次 DSL context 的 chip 数 |
| `core_num` | 兼容参数；实际 launch 时通过 grid / launch config 选择 core 数 |

注意事项：

- 在启动 Kernel 前调用
- 重新调用会重置当前 DSL context
- 多芯片 Kernel 的 `chip_num` 决定记录多少个 chip graph
- 某些 run mode / 后端对多芯片支持有限制，见后端文档

## 3. `@ax.builder`

常用形式：

```python
@ax.builder
def kernel(...):
    ...
```

带配置：

```python
@ax.builder(
    run_mode=ax.RunMode.CRUN,
    enable_cache=False,
    check=ax.Check.ALL_CLOSE,
    check_mode=ax.CheckMode.CHECK_PER_LAYER,
)
def kernel(...):
    ...
```

常用参数：

| 参数 | 说明 |
|---|---|
| `run_mode` | PYRUN / CRUN / BRUN |
| `backend` | 默认 AX8860，也可选择支持的 backend |
| `enable_cache` | 是否启用编译缓存 |
| `check` | CRUN 与 PYRUN 的对比策略 |
| `check_mode` | 输出检查或逐层检查 |
| `check_rtol` | 相对误差 |
| `check_atol` | 绝对误差 |
| `check_cosine_similarity` | 余弦相似度阈值 |
| `const_inputs` | 显式指定哪些输入按编译期常量处理 |
| `dump_dsl_perf` | 导出性能分析相关材料 |
| `dump_path` | 调试与编译产物目录 |
| `cache_path` | 缓存目录，默认 `~/.cache/axdsl` |

`check` 与 `enable_cache` 不能同时开启，因为缓存中不保留 correctness check 所需的全部调试信息。

## 4. Kernel 启动与 Grid

### 一维 grid

```python
kernel[8](...)
```

表示 8 个 program，`ax.program_id(0)` 取值 0 到 7。

### 二维 grid 与 core 数

推荐显式写出 grid 和 core 数：

```python
kernel[(grid_m, grid_n), core_num](...)
```

方括号中的 `(grid_shape, core_num)` 是 launch config。二维 tile 中：

- `program_id(0)` 表示 M 方向 tile 编号
- `program_id(1)` 表示 N 方向 tile 编号

如果省略 `core_num`，默认值为 `grid_m * grid_n`；多维 grid 建议显式指定，降低 launch config 歧义。实际可用 core 数受 AX8860 每 chip core 数和后端调度限制。

## 5. Program 与数据所有权

Grid 只定义逻辑任务数量，不自动知道 tensor 应该如何切分。Kernel 作者必须建立映射。

### 按行切分

```python
row = ax.program_id(0)
x_tile = x[row * TILE_M:(row + 1) * TILE_M, :]
```

### 二维 tile

```python
tile_m = ax.program_id(0)
tile_n = ax.program_id(1)
x_tile = x[tile_m * TILE_M:(tile_m + 1) * TILE_M, :]
w_tile = w[:, tile_n * TILE_N:(tile_n + 1) * TILE_N]
```

### 使用总 program 数

```python
num = ax.num_programs(0)
```

适合 work partition 与尾块处理。

## 6. 参数类型

### Runtime tensor

Torch tensor 传入后，在 Kernel 内成为 DSL tensor：

```python
def kernel(x, output, tile_size):
```

特点：

- shape / dtype 可见
- 元素值不可见
- 不同 shape / dtype 会影响编译缓存 key
- 可用于 `load`、compute、`store`

### Compile-time constant

普通 Python 对象按编译期常量处理：

```python
kernel[grid](x, output, 128)
```

常见类型：

- int
- bool
- tuple
- list
- NumPy array
- Python 函数对象

它们可以参与 Python 分支、切片和循环，但值变化会生成不同的编译配置。

### `const_inputs`

如果需要强制某些输入按常量处理，可在 builder 上声明：

```python
@ax.builder(const_inputs=["weight"])
def kernel(x, weight, output):
    ...
```

适合权重、静态 lookup table 等数据。

## 7. Tensor 切片规则

### 推荐：rank-preserving slice

一维：

```python
x[start:stop]
```

二维：

```python
x[m0:m1, n0:n1]
```

### 不推荐 / 不支持：降 rank 的整数索引

```python
x[i]
```

应改为：

```python
x[i:i+1]
```

### Python slice 对象也可以

```python
tile = slice(start, stop)
x_tile = x[tile]
```

### 尾部 tile

始终使用 `min` 或等价逻辑：

```python
stop = min(start + tile_size, x.shape[0])
```

## 8. 写输出

### 用 `store` 写 DDR

```python
ax.dsl.store(x=result, r=output[start:stop])
```

### 用算子的 `r` 参数写目标

部分 API 支持 `r`：

```python
ax.dsl.add(x, y, r=temp)
ax.dsl.sum(x, dim=-1, r=out_slice)
```

### 不支持的 partial assignment

```python
output[start:stop] = result
```

应改为：

```python
ax.dsl.move(result, r=output[start:stop])
# 或在产生 result 的算子上直接传 r
```

### 多输出

```python
return output_a, output_b
```

## 9. Kernel 返回值

最后应返回输出 tensor 或输出 tuple：

```python
def kernel(x, out_a, out_b):
    ...
    return out_a, out_b
```

返回值用于：

- 确定输出列表
- 构建 graph output contract
- PyRun 写回结果
- CRUN 正确性对比
- 与推理引擎集成

## 10. Python 语法支持速查

| 写法 | 支持 | 说明 |
|---|---|---|
| 普通 `if` 常量条件 | 支持 | 编译期确定分支 |
| 普通 `for` 常量范围 | 支持 | trace 时展开 |
| tensor 值作为 `if` 条件 | 不支持 | 使用 `ax.dsl.if_op` |
| tensor 值作为循环终止条件 | 不支持 | 使用 `ax.dsl.while_loop` |
| `x.shape` | 支持 | 读取元数据 |
| `x` 元素值 | 不支持 | 使用算子表达 |
| `x[i:i+n]` | 支持 | 保持 rank |
| `x[i]` | 不支持 | 改为 `x[i:i+1]` |
| `output[i:i+n] = y` | 不建议 / 受限 | 使用 `r=` 或 `store` |
| Python 函数复用 | 支持 | 会被 trace 进 Kernel |
| 任意第三方库调用 | 不保证支持 | 只依赖官方文档允许的库 |

## 11. Kernel 组织建议

### 11.1 Kernel 只表达设备逻辑

避免在 Kernel 内做 I/O、下载、日志、随机初始化等 host 逻辑。

### 11.2 用 wrapper 封装 grid

```python
def run(x, output):
    tile_size = choose_tile(x)
    grid = (x.shape[0] + tile_size - 1) // tile_size
    kernel[grid](x, output, tile_size)
```

好处：

- 用户只需要关心输入输出
- tile / grid 策略集中管理
- 方便做 shape 特化

### 11.3 reference 与 Kernel 分离

```python
def reference(x):
    return x * 2

def test():
    ...
```

便于 PYRUN、CRUN 和 PyTorch 三方对照。

### 11.4 小 shape 先跑通

推荐顺序：

```text
1 × tile
非整除 shape
多个 program
多个 core
多 chip
```

### 11.5 输出显式初始化或显式写满

`torch.empty_like` 可用于输出 buffer，但必须确保 Kernel 每个位置都被写入。否则 `torch.testing.assert_close` 可能读到未定义值。

## 12. 最小 checklist

写一个新 Kernel 前，可以先回答：

- 输入输出 dtype 和 shape 是什么？
- 输出由谁分配？
- grid 如何映射 tensor 区域？
- 是否有尾块？
- tile 是否超过 OCM？
- 哪些参数是编译期常量？
- 是否所有输出都被写满？
- reference 是什么？
- tolerance 是多少？
- 用 PYRUN 还是 CRUN 验证？
- 是否需要 graph / trace dump？

这些问题的答案会直接决定 Kernel 是否容易调试和优化。
