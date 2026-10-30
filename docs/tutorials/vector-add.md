# Vector Add：第一个 Kernel

这个教程把快速入门的 Vector Add 完整走一遍，重点理解 Grid、program 坐标、尾块和 DDR → OCM → DDR 数据流。

## 1. 目标

实现：

```text
output[i] = x[i] + y[i]
```

要求：

- 输入长度可以不被 tile size 整除
- 每个 program 只处理一个 tile
- 输出由 caller 预分配
- 用 PyTorch reference 验证

## 2. 问题切分

选择：

```text
TILE_SIZE = 128
N = 1000
```

需要的 program 数：

```text
grid = ceil(1000 / 128) = 8
```

每个 program 的区间：

| program | start | stop | length |
|---|---:|---:|---:|
| 0 | 0 | 128 | 128 |
| 1 | 128 | 256 | 128 |
| 2 | 256 | 384 | 128 |
| 3 | 384 | 512 | 128 |
| 4 | 512 | 640 | 128 |
| 5 | 640 | 768 | 128 |
| 6 | 768 | 896 | 128 |
| 7 | 896 | min(1024, 1000) | 104 |

这就是 `min(start + TILE_SIZE, x.shape[0])` 的作用。

## 3. Kernel 实现

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
```

## 4. 启动 wrapper

```python
def vector_add_wrapper(x, y, output):
    TILE_SIZE = 128
    grid = (x.shape[0] + TILE_SIZE - 1) // TILE_SIZE
    vector_add_kernel[grid](x, y, output, TILE_SIZE)
```

wrapper 的职责：

- 选择 tile size
- 计算 grid
- 传 runtime tensor 和 compile-time tile size
- 隐藏 launch 细节

## 5. 验证

```python
def main():
    ax.init(chip_num=1, core_num=1)

    x = torch.arange(1000, dtype=torch.float32)
    y = torch.arange(1000, dtype=torch.float32) * 2
    output_ax = torch.empty_like(x)

    output_gt = x + y
    vector_add_wrapper(x, y, output_ax)

    torch.testing.assert_close(output_gt, output_ax)
    print("AXDSL vector add: PASS")
```

## 6. 数据流图

```text
Program 0
  x[0:128] DDR     y[0:128] DDR
        │                │
        └── load ───────┘
                 ↓
              OCM tiles
                 ↓
             ax.dsl.add
                 ↓
             OCM result
                 ↓
           store to output[0:128] DDR

Program 1
  x[128:256] + y[128:256] → output[128:256]
...

Program 7
  x[896:1000] + y[896:1000] → output[896:1000]
```

## 7. 运行

```bash
source script/npu_dev
python3 axnn/axnn/dsl/user_docs/examples/vector_add.py
```

或者运行本站附带示例：

```bash
python3 docs/examples/vector_add.py
```

## 8. 练习

### 练习 1：修改 tile size

把 `TILE_SIZE` 分别改成 16、64、256，观察 grid 数量是否正确。

### 练习 2：处理其他长度

把输入长度改为：

```python
1, 127, 128, 129, 4096
```

确认仍然通过。

### 练习 3：改成乘法

把 `add` 改成 `mul`，并同步修改 reference。

### 练习 4：添加第三个输入

实现：

```python
output[i] = x[i] + y[i] + z[i]
```

### 练习 5：CRUN 验证

在工具链可用时，把 builder 改为：

```python
@ax.builder(
    run_mode=ax.RunMode.CRUN,
    enable_cache=False,
    check=ax.Check.ALL_CLOSE,
)
```

## 9. 常见问题

### 为什么不改 output 后也 return

`return output` 用于明确 Kernel 输出契约，是 builder 识别输出的重要信息。

### 为什么不用 Python for 遍历元素

AXDSL 面向 tensor / tile 计算。逐元素 Python 循环会生成大量 IR，不利于 NPU 执行。

### 为什么每个 program 只处理一个 tile

这是最简单、清晰的映射。后续可以扩展为一个 program 内部再循环多个 tile。

## 10. 下一步

进入 [Tiled MatMul](/tutorials/tiled-matmul)，学习二维 grid 和 tile 复用。
