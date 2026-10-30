# 常用算子与 PyTorch 迁移

本章按“能力类别”介绍常用 AXDSL 算子，帮助你从 PyTorch 表达迁移到 AXDSL。完整签名和所有算子见 [完整算子列表](/reference/operators)。

## 1. 使用原则

### 1.1 先语义，再低精度

推荐迁移顺序：

```text
PyTorch fp32 reference
  ↓
AXDSL fp32 / PYRUN
  ↓
AXDSL CRUN
  ↓
bf16 / fp16 / int8
  ↓
tile 与性能优化
```

### 1.2 不要在 Kernel 内读取 tensor 值

```python
# 错误：运行期值不能转回 Python 判断
if x.numpy().sum() > 0:
    ...

# 正确：用 DSL 算子表达
y = ax.dsl.sum(x, dim=-1)
```

### 1.3 中间结果尽量留在 DSL 图内

```python
a = ax.dsl.load(x)
b = ax.dsl.silu(a)
c = ax.dsl.mul(b, a)
ax.dsl.store(x=c, r=output)
```

不要为了“调试方便”把每个中间结果都写回 DDR，除非确实需要观察。

## 2. 能力矩阵

| 类别 | 常用 API | 对应 PyTorch / 语义 |
|---|---|---|
| Elementwise | `add`, `sub`, `mul`, `div` | `+`, `-`, `*`, `/` |
| Activation | `sigmoid`, `tanh`, `gelu`, `quick_gelu`, `silu`, `softplus` | 常见激活函数 |
| Unary | `exp`, `log2`, `sqrt`, `reciprocal`, `floor`, `ceil` | 数学函数 |
| Reduction | `sum`, `max`, `min`, `argmax`, `argmin`, `cumsum` | reduce / scan |
| Dense | `matmul`, `conv` | `matmul` / convolution |
| Shape | `reshape`, `transpose`, `concat`, `expand`, `pad` | shape 变换 |
| Index | `gather`, `scatter`, `index_select`, `index_add`, `select_scatter` | 索引读写 |
| Search | `topk`, `nonzero` | top-k / 非零位置 |
| Select | `where`, `maximum`, `minimum`, `clip` | 条件选择与裁剪 |
| Cast | `cast` | dtype 转换 |
| Creation | `zeros`, `full`, `from_numpy`, `create_tensor` | 常量 / tensor 创建 |
| Memory | `load`, `store`, `move`, `dynamic_ld`, `dynamic_st` | 数据搬运 |
| Debug | `device_print` | 设备端调试 |

## 3. Elementwise

### 基础四则

```python
a = ax.dsl.add(x, y)
b = ax.dsl.sub(x, y)
c = ax.dsl.mul(x, y)
d = ax.dsl.div(x, y)
```

与 PyTorch 对照：

```python
a = x + y
b = x - y
c = x * y
d = x / y
```

### 写入指定 tensor

```python
temp = ax.dsl.create_tensor(shape=x.shape, dtype=x.dtype)
ax.dsl.add(x, y, r=temp)
```

### 融合机会

连续 elementwise 通常容易被图优化器融合，例如：

```python
y = ax.dsl.mul(ax.dsl.add(x, one), scale)
```

可减少中间数据落存。

## 4. Activation

| AXDSL | 语义 |
|---|---|
| `sigmoid(x)` | `1 / (1 + exp(-x))` |
| `tanh(x)` | 双曲正切 |
| `gelu(x)` | GELU |
| `quick_gelu(x)` | Quick GELU |
| `silu(x)` | `x * sigmoid(x)` |
| `softplus(x)` | Softplus |

示例 SwiGLU 片段：

```python
gate = ax.dsl.silu(x)
y = ax.dsl.mul(gate, value)
```

对应 PyTorch：

```python
y = torch.nn.functional.silu(x) * value
```

## 5. Unary math

```python
y_exp = ax.dsl.exp(x)
y_log2 = ax.dsl.log2(x)
y_sqrt = ax.dsl.sqrt(x)
y_reciprocal = ax.dsl.reciprocal(x)
y_floor = ax.dsl.floor(x)
y_ceil = ax.dsl.ceil(x)
```

注意：

- 低精度下 `exp`、`log2` 的误差策略可能不同
- 除法可用 `reciprocal` 乘法替代以匹配后端优化，但应先验证精度
- `ceil` / `floor` 的输入输出 dtype 应按 API 支持矩阵确认

## 6. Reduction

### sum

```python
s = ax.dsl.sum(x, dim=-1, keepdim=True)
```

对应：

```python
s = x.sum(dim=-1, keepdim=True)
```

### max / min

```python
m = ax.dsl.max(x, dim=-1, keepdim=True)
v = ax.dsl.min(x, dim=-1, keepdim=True)
```

### argmax / argmin

```python
idx = ax.dsl.argmax(x, dim=-1)
```

### cumsum

```python
y = ax.dsl.cumsum(x, dim=-1)
```

### softmax

```python
p = ax.dsl.softmax(x, dim=-1)
```

### 数值稳定性

如果手写 softmax，应参考稳定形式：

```python
m = ax.dsl.max(x, dim=-1, keepdim=True)
shifted = ax.dsl.sub(x, m)
e = ax.dsl.exp(shifted)
s = ax.dsl.sum(e, dim=-1, keepdim=True)
y = ax.dsl.div(e, s)
```

低精度场景下，中间累计可能需要更高 output dtype，再在最终阶段 cast 回目标 dtype。

## 7. Matmul

基本用法：

```python
y = ax.dsl.matmul(x, w, transposed=True)
```

当 `w` 已经是 `(N, K)` 且 `x` 是 `(..., K)` 时，`transposed=True` 表示执行：

```python
y = x @ w.transpose(-1, -2)
```

### 常见 shape

```text
x: (M, K)
w: (N, K), transposed=True
y: (M, N)
```

### 常量权重

如果权重可视为编译期常量，可通过 `const_inputs` 或相应 API 参数传递，让后端进行更充分的优化。示例见仓库 `demo_matmul.py`。

### 精度

常见 dtype：

- fp32：语义验证
- bf16 / fp16：LLM 与常见推理
- int8 / fp8 / scaled dtype：需确认后端与 scale 语义

### tile

Matmul 是最能体现 AXDSL 价值的算子之一。二维 grid 常见映射：

```text
program(pid_m, pid_n)
  A[pid_m * TM : (pid_m+1)*TM, :]
  B[:, pid_n * TN : (pid_n+1)*TN]
  C[pid_m * TM : (pid_m+1)*TM,
    pid_n * TN : (pid_n+1)*TN]
```

完整教程见 [Tiled MatMul](/tutorials/tiled-matmul)。

## 8. Convolution

```python
y = ax.dsl.conv(
    x,
    w,
    stride=(1, 1),
    padding=(1, 1, 1, 1),
    layout="NHWC",
)
```

支持 layout 与权重形式以工具链为准。对于视觉模型迁移，建议先确认：

- layout：NCHW / NHWC
- weight layout
- group / dilation
- 常量权重或 runtime 权重
- 量化 scale

## 9. Shape 操作

### reshape

```python
y = ax.dsl.reshape(x, (M, N))
```

reshape 只改变视图 / IR 形状，不搬运数据语义上的元素重排。实际后端仍可能有 layout 要求。

### transpose

```python
y = ax.dsl.transpose(x, perm=(1, 0, 2))
```

### concat

```python
y = ax.dsl.concat((a, b), dim=0)
```

### expand

```python
y = ax.dsl.expand(x, shape=(B, M, N))
```

### pad

```python
y = ax.dsl.pad(x, padding=(0, 1, 0, 0))
```

## 10. Index 操作

### gather

```python
y = ax.dsl.gather(x, indices, dim=0)
```

按 index 从 `x` 中收集数据。

### scatter

```python
ax.dsl.scatter(r=output, indices=indices, x=values, dim=0)
```

### index_select

```python
y = ax.dsl.index_select(x, indices, dim=0)
```

### index_add

```python
ax.dsl.index_add(x=values, indices=indices, r=output, dim=0)
```

适合 MoE combine、加权累加等场景。

### select_scatter

将源数据按索引写入目标 tensor 的部分区域。具体签名见 API 参考。

### topk / nonzero

```python
values, indices = ax.dsl.topk(x, k=8, dim=-1)
indices, count = ax.dsl.nonzero(mask)
```

适合 router、稀疏选择、动态 token 过滤等场景。返回契约与 dtype 需按 API 文档确认。

## 11. 条件与裁剪

```python
y = ax.dsl.where(cond, a, b)
z = ax.dsl.maximum(x, zero)
w = ax.dsl.minimum(x, limit)
c = ax.dsl.clip(x, min_value, max_value)
```

ReLU 可表达为：

```python
zero = ax.dsl.zeros(shape=x.shape, dtype=x.dtype)
y = ax.dsl.maximum(x, zero)
```

## 12. dtype 与 cast

常见别名：

- `ax.fp32`
- `ax.bf16`
- `ax.fp16`
- `ax.fp8_e4m3`
- `ax.int8`
- `ax.uint8`
- `ax.int32`
- `ax.uint32`

转换：

```python
y = ax.dsl.cast(x, dtype=ax.bf16)
```

一些算子支持 `output_dtype`：

```python
s = ax.dsl.sum(x, dim=-1, output_dtype=ax.fp32)
```

建议：

- reduce 前半段保留更高精度
- 最终输出再 cast
- reference 也模拟必要的 dtype 转换

## 13. 常量与创建

### zeros

```python
zero = ax.dsl.zeros(shape=x.shape, dtype=x.dtype)
```

### full

```python
one = ax.dsl.full(shape=(1,), fill_value=1, dtype=ax.int32)
```

### from_numpy

```python
import numpy as np
step = ax.dsl.from_numpy(np.array([1], dtype=np.int32), prefix="step")
```

适合控制流中的 INT32 标量 tensor。

### create_tensor

```python
temp = ax.dsl.create_tensor(shape=x.shape, dtype=x.dtype)
```

适合显式声明中间输出。

## 14. PyTorch 到 AXDSL 速查

| PyTorch | AXDSL |
|---|---|
| `x + y` | `ax.dsl.add(x, y)` |
| `x - y` | `ax.dsl.sub(x, y)` |
| `x * y` | `ax.dsl.mul(x, y)` |
| `x / y` | `ax.dsl.div(x, y)` |
| `torch.max(x, y)` | `ax.dsl.maximum(x, y)` |
| `torch.min(x, y)` | `ax.dsl.minimum(x, y)` |
| `x.sum(dim=-1)` | `ax.dsl.sum(x, dim=-1)` |
| `x.argmax(dim=-1)` | `ax.dsl.argmax(x, dim=-1)` |
| `x @ w.T` | `ax.dsl.matmul(x, w, transposed=True)` |
| `F.softmax(x, dim=-1)` | `ax.dsl.softmax(x, dim=-1)` |
| `F.silu(x)` | `ax.dsl.silu(x)` |
| `x.reshape(...)` | `ax.dsl.reshape(x, ...)` |
| `x.transpose(...)` | `ax.dsl.transpose(x, ...)` |
| `torch.cat(...)` | `ax.dsl.concat(...)` |
| `x.to(torch.bfloat16)` | `ax.dsl.cast(x, dtype=ax.bf16)` |
| `x[index]` | `ax.dsl.index_select(...)` / `gather(...)` |
| `out.index_add_(...)` | `ax.dsl.index_add(..., r=out)` |

## 15. 迁移 checklist

把一个 PyTorch 函数迁移为 AXDSL 时，建议检查：

- [ ] 输入输出 shape 和 dtype
- [ ] 是否需要 caller-allocated output
- [ ] 哪些参数应作为常量
- [ ] tile 粒度
- [ ] grid 映射
- [ ] 是否有尾块
- [ ] 中间结果是否需要落 DDR
- [ ] 是否有 runtime 条件分支
- [ ] 是否有动态索引
- [ ] reduce 的数值稳定性
- [ ] reference 与 tolerance
- [ ] 后端 dtype 支持
