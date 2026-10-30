# 10 分钟快速上手

本章目标是用最小例子跑通 AXDSL，并理解每个关键语句的含义。示例不需要先掌握多核、多芯片和控制流，只需要理解：

- Kernel 定义
- Grid 启动
- Program 坐标
- DDR → OCM → DDR 的数据流
- 输出显式返回

## 1. 环境准备

AXDSL 包含在 AXNN Python 包中。你需要：

- 当前仓库版本支持的 Python 环境
- 可 import 的 `axnn` 包
- 仓库依赖已安装
- 如需 CRUN / BRUN，还需对应 AX8860 NPU 工具链

在仓库根目录进入开发环境：

```bash
source script/npu_dev
```

确认当前 Python 能找到包：

```bash
python3 -c "import axnn.dsl as ax; print(ax)"
```

::: warning
不同 release 支持的芯片、backend 与 dtype 可能不同。正式教程发布时应替换为目标工具链的安装方式与版本要求。
:::

## 2. 运行第一个 Kernel

官方最小示例：

```bash
python3 axnn/axnn/dsl/user_docs/examples/vector_add.py
```

预期输出：

```text
AXDSL vector add: PASS
```

完整代码如下：

```python
import torch

import axnn.dsl as ax


@ax.builder(run_mode=ax.RunMode.PYRUN)
def vector_add_kernel(x, y, output, tile_size):
    start = ax.program_id(0) * tile_size
    stop = min(start + tile_size, x.shape[0])
    tile = slice(start, stop)

    x_tile = ax.dsl.load(x[tile])
    y_tile = ax.dsl.load(y[tile])
    result = ax.dsl.add(x_tile, y_tile)
    ax.dsl.store(x=result, r=output[tile])
    return output


def vector_add_wrapper(x, y, output):
    TILE_SIZE = 128
    grid = (x.shape[0] + TILE_SIZE - 1) // TILE_SIZE
    vector_add_kernel[grid](x, y, output, TILE_SIZE)


def main():
    ax.init(chip_num=1, core_num=1)
    x = torch.arange(1000, dtype=torch.float32)
    y = torch.arange(1000, dtype=torch.float32) * 2
    output_ax = torch.empty_like(x)

    output_gt = x + y
    vector_add_wrapper(x, y, output_ax)

    torch.testing.assert_close(output_gt, output_ax)
    print("AXDSL vector add: PASS")


if __name__ == "__main__":
    main()
```

## 3. 逐行理解

### 3.1 初始化上下文

```python
ax.init(chip_num=1, core_num=1)
```

`ax.init` 创建 / 重置当前 DSL recording context，并声明参与 Kernel 的 chip 数。`core_num` 参数当前用于兼容，实际每次 launch 会通过 grid / launch config 选择 core 数。

重新调用 `ax.init` 会重置当前配置。

### 3.2 定义 Kernel

```python
@ax.builder(run_mode=ax.RunMode.PYRUN)
def vector_add_kernel(...):
    ...
```

`@ax.builder` 把 Python 函数包装为 launchable kernel。`run_mode=ax.RunMode.PYRUN` 表示先用 Python reference 实现执行，适合第一步验证语义。

### 3.3 计算 Program 负责的数据段

```python
start = ax.program_id(0) * tile_size
stop = min(start + tile_size, x.shape[0])
```

`ax.program_id(0)` 是当前 program 在 grid 第 0 维的坐标。如果 `grid=8`，每个 program 的坐标分别是 0 到 7。

`min` 处理最后一个不完整 tile。示例输入长度是 1000，tile size 是 128：

```text
1000 = 7 × 128 + 104
```

因此第 7 个 program 只处理 104 个元素。

### 3.4 从 DDR 载入 tile

```python
x_tile = ax.dsl.load(x[tile])
y_tile = ax.dsl.load(y[tile])
```

`x` 与 `y` 是 caller 传入的 DDR tensor。`load` 表示把切片数据搬入 core 可用的局部计算内存，通常是 OCM。

### 3.5 在局部 tile 上计算

```python
result = ax.dsl.add(x_tile, y_tile)
```

这里不是直接修改 `x_tile` 或 `y_tile`，而是记录一个 `add` 操作并返回结果 tensor。

### 3.6 写回输出

```python
ax.dsl.store(x=result, r=output[tile])
```

AXDSL Kernel 的输出通常由 caller 预分配。`store` 把 OCM 中的结果写回 DDR 输出切片。

### 3.7 返回输出

```python
return output
```

返回语句用于帮助编译器识别 Kernel 输出。多个输出时可以返回 tuple。

## 4. 启动语法

```python
grid = (x.shape[0] + TILE_SIZE - 1) // TILE_SIZE
vector_add_kernel[grid](x, y, output, TILE_SIZE)
```

含义：

1. `vector_add_kernel[grid]` 设置逻辑 program grid。
2. `(...)` 传入 runtime 参数和编译期参数。
3. AXDSL 记录并执行这个 Kernel。

一维 grid 可以写整数：

```python
kernel[8](...)
```

二维或多维 grid 推荐和 core 数一起显式写出：

```python
kernel[(grid_m, grid_n), core_num](...)
```

这里方括号中的整体是 launch config：第一项 `grid_shape`，第二项 `core_num`。省略 `core_num` 时默认为 grid 元素数量；但对多维 grid，建议始终显式指定，避免把 `(grid_m, grid_n)` 误解成其他 launch config 形式。

## 5. 参数语义速览

| 参数类型 | Kernel 内语义 | 示例 |
|---|---|---|
| Torch tensor / DSL tensor | 运行期 tensor，值不能在编译期读取 | `x`, `y`, `output` |
| int / tuple / list | 编译期常量 | `tile_size` |
| NumPy 常量 | 默认按常量处理，或通过 `const_inputs` 显式声明 | weight |

因此下面代码中的 `tile_size` 可以直接用于 Python 表达式：

```python
start = ax.program_id(0) * tile_size
```

而 `x.shape[0]` 中的 shape 是元数据，允许编译期读取；`x` 的元素值不能读取。

## 6. 正确性验证

始终建议先写 reference：

```python
output_gt = x + y
vector_add_wrapper(x, y, output_ax)
torch.testing.assert_close(output_gt, output_ax)
```

选择 tolerance 时应考虑 dtype：

| dtype | 建议 |
|---|---|
| fp32 | 可使用较小 rtol / atol |
| bf16 / fp16 | 应容忍较大误差 |
| int | 精确比较 |
| 低精度浮点 / 量化类型 | 使用任务相关指标，不只看逐点误差 |

## 7. 常见第一次失败原因

### Shape 不整除 tile

错误表现：越界、输出部分未写、结果错误。

解决：

```python
stop = min(start + tile_size, x.shape[0])
```

### 忘记处理输出

错误表现：输出 tensor 没有被写满。

解决：确保所有输出切片都被 `store` 或其他算子写入，并返回 output。

### 使用了不支持的切片

AXDSL 只支持保持 rank 的切片，例如：

```python
x[i:i + tile]
```

不支持把 rank 降一的整数索引：

```python
x[i]
```

需要改成：

```python
x[i:i+1]
```

### 尝试普通赋值写结果

不推荐也不支持所有场景的 partial assignment：

```python
y[i:i+1] = ax.dsl.some_op(x)
```

应使用输出参数：

```python
ax.dsl.some_op(x, r=y[i:i+1])
```

### PYRUN 正确但 CRUN 失败

常见原因：

- dtype 超出后端支持
- shape / layout 不满足算子限制
- 控制流状态更新不符合 API 契约
- 并行同步顺序错误
- tile 超出 OCM

建议先缩小 shape，再开启逐层检查定位。

## 8. 下一步

- 深入执行链路：[总体架构](/guide/architecture)
- 掌握 Kernel 规则：[编程模型](/guide/programming-model)
- 理解 DDR / OCM：[内存模型](/guide/memory-model)
- 完整走一遍示例：[Vector Add 教程](/tutorials/vector-add)
