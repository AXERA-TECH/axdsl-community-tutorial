# Kernel 启动与离线编译

本章对应官方 20261116 文档新增的 Kernel Launch / Kernel Compile 说明，介绍：

- launch 方括号里的 grid shape 和 core 数如何解析
- 多维 grid 的 trailing comma 写法
- 如何用 `compile()` 生成 `.compiled.axmodel`
- 如何把编译产物复制到开发板并让 BRUN 命中缓存

## 1. Launch 语法

基本形式：

```python
kernel[grid_shape](*args, **kwargs)
kernel[grid_shape, core_num](*args, **kwargs)
```

- `grid_shape`
  - 一维 grid 时是正整数
  - 多维 grid 时是非空 tuple
- `core_num`
  - 可选正整数
  - 省略时默认等于所有 grid 维度的乘积
- `(...)` 传入 Kernel 参数

## 2. 一维 grid

```python
kernel[4](a, output)
```

含义：

```text
grid shape = 4
program 数 = 4
core_num 默认 = 4
```

如果希望 4 个 program 由 1 个 core 执行：

```python
kernel[4, 1](a, output)
```

含义：

```text
grid shape = 4
program 数 = 4
core_num = 1
```

## 3. 多维 grid

二维 grid 有 4 个 program：

```python
kernel[(2, 2),](a, output)
```

含义：

```text
grid shape = (2, 2)
program 数 = 4
core_num 默认 = 4
```

这里的 trailing comma 非常重要：

```python
kernel[(2, 2),]
```

表示第一个方括号参数是 tuple `(2, 2)`。

如果写成：

```python
kernel[2, 2]
```

含义完全不同：

```text
grid shape = 2
core_num = 2
```

也就是一维 2 program grid，而不是二维 `(2, 2)` grid。

## 4. 常用 launch 写法速查

| 写法 | grid shape | program 数 | core 数 |
|---|---|---:|---:|
| `kernel[4]` | `4` | 4 | 4 |
| `kernel[4, 1]` | `4` | 4 | 1 |
| `kernel[(2, 2),]` | `(2, 2)` | 4 | 4 |
| `kernel[(2, 2), 1]` | `(2, 2)` | 4 | 1 |
| `kernel[grid,]`，其中 `grid=(2, 2)` | `(2, 2)` | 4 | 4 |
| `kernel[grid, 1]`，其中 `grid=(2, 2)` | `(2, 2)` | 4 | 1 |

推荐两种风格：

1. **字面量二维 grid**

   ```python
   kernel[(2, 2), 1](...)
   ```

2. **变量保存 tuple**

   ```python
   grid = (2, 2)
   kernel[grid, 1](...)
   ```

如果确实希望使用默认 core 数，也要保留 tuple 边界：

```python
grid = (2, 2)
kernel[grid,](...)
```

## 5. 为什么 trailing comma 重要

Python 方括号语法中：

```python
kernel[2, 2]
```

等价于向 `__getitem__` 传入 tuple：

```python
(2, 2)
```

AXDSL 把这个 tuple 解析为：

```text
第一项：grid shape = 2
第二项：core_num = 2
```

而：

```python
kernel[(2, 2),]
```

传入的是：

```python
((2, 2),)
```

AXDSL 解析为：

```text
第一项：grid shape = (2, 2)
第二项：core_num 未指定，使用默认值 4
```

因此，多维 grid 使用默认 core 数时，必须写成：

```python
kernel[(2, 2),]
```

或：

```python
grid = (2, 2)
kernel[grid,]
```

## 6. `compile()` 离线编译

`compile()` 可以只编译 Kernel，不启动它：

```python
compiled_path = kernel[grid_shape, core_num].compile(*args, **kwargs)
```

返回值是 `pathlib.Path`，指向：

```text
<cache_path>/<kernel_name>.<hash>.compiled.axmodel
```

默认 `cache_path`：

```text
~/.cache/axdsl
```

也可以在 `@ax.builder` 中指定：

```python
@ax.builder(
    run_mode=ax.RunMode.BRUN,
    enable_cache=True,
    cache_path="/path/to/cache",
)
def kernel(...):
    ...
```

## 7. 编译示例

构建机器上的 `kernel_compile_demo.py`：

```python
import torch

import axnn.dsl as ax


@ax.builder(
    run_mode=ax.RunMode.BRUN,
    enable_cache=True,
)
def add_kernel(a, b, output):
    lhs = ax.dsl.load(a)
    rhs = ax.dsl.load(b)
    ax.dsl.store(ax.dsl.add(lhs, rhs), r=output)
    return output


ax.init(chip_num=1)

a = torch.ones(16, dtype=torch.float32)
b = torch.full_like(a, 2)
output = torch.empty_like(a)

compiled_path = add_kernel[1, 1].compile(a, b, output)
print(compiled_path)
```

运行：

```bash
python3 kernel_compile_demo.py
```

可能输出：

```text
/home/<user>/.cache/axdsl/add_kernel.<hash>.compiled.axmodel
```

## 8. 复制编译产物到开发板

假设开发板登录名为 `user`，地址为 `board`：

```bash
ssh user@board 'mkdir -p ~/.cache/axdsl'

scp ~/.cache/axdsl/add_kernel.*.compiled.axmodel \
    user@board:~/.cache/axdsl/
```

板端程序使用相同 cache path，并在 BRUN 中保持：

```python
@ax.builder(
    run_mode=ax.RunMode.BRUN,
    enable_cache=True,
)
def add_kernel(...):
    ...
```

当 cache key 匹配时，BRUN 直接加载已编译的 AXModel，不再重新编译。

## 9. cache key 必须匹配

板端 BRUN 想命中同一个 `.compiled.axmodel`，构建机器和板端需要保持一致的属性包括：

- AXNN toolchain 版本
- Kernel 源码 / AST
- launch grid shape
- core 数
- chip 数
- 输入 shape
- 输入 dtype
- 输出 shape
- 输出 dtype
- compile-time 参数值
- 编译配置

任意一项不同，都可能生成不同 cache key，导致 cache miss 并重新编译。

## 10. 常见问题

### `kernel[2, 2]` 为什么不是二维 grid？

因为 AXDSL 将方括号里的 `(2, 2)` 解析为：

```text
grid_shape = 2
core_num = 2
```

二维 grid 应写：

```python
kernel[(2, 2),]
```

或：

```python
kernel[(2, 2), 1]
```

### 已经指定 `enable_cache=True`，为什么板端还是重新编译？

优先检查 cache key：

```text
工具链版本是否一致？
kernel 源码是否一致？
grid / core / chip 配置是否一致？
输入输出 shape 和 dtype 是否一致？
compile-time 参数是否一致？
cache 文件是否复制到板端 cache_path？
```

### `compile()` 会执行 Kernel 吗？

不会。`compile()` 只记录、编译并导出 AXModel，不启动推理。

### 可以编译后直接用 `pulsar2 run` 吗？

可以参考目标工具链的 AXModel 运行方式。AXDSL 官方示例给出的板端路径是通过 BRUN 和 AXCL 加载编译产物；`pulsar2 run` 的输入输出目录格式以当前工具链说明为准。

## 11. 小结

```text
kernel[4]             # 1D grid，默认 4 core
kernel[4, 1]          # 1D grid，1 core
kernel[(2, 2),]       # 2D grid，默认 4 core
kernel[(2, 2), 1]     # 2D grid，1 core
```

离线编译流程：

```text
构建机器 compile()
  ↓
生成 <kernel>.<hash>.compiled.axmodel
  ↓
scp 到板端 cache_path
  ↓
板端 BRUN + enable_cache=True
  ↓
cache key 匹配后直接加载执行
```
