# Fused SiLU × Mul：算子融合

LLM MLP 中常见结构：

```text
y = SiLU(gate) * value
```

分开实现会引入中间结果：

```text
gate → SiLU → activated
activated × value → y
```

如果 `activated` 被写入 DDR 再读回，会增加额外 traffic。AXDSL 可以把两个操作记录在同一 Graph 中，让后端有机会融合。

## 1. 目标

实现：

```python
def reference(gate, value):
    return torch.nn.functional.silu(gate) * value
```

要求：

- tile 化处理
- 支持长度不整除
- 中间 `activated` 保持在 DSL 图内
- 输出 caller 分配

## 2. Kernel

```python
@ax.builder(run_mode=ax.RunMode.PYRUN)
def fused_silu_mul_kernel(gate, value, output, tile_size):
    start = ax.program_id(0) * tile_size
    stop = min(start + tile_size, gate.shape[0])

    gate_tile = ax.dsl.load(gate[start:stop])
    value_tile = ax.dsl.load(value[start:stop])

    activated = ax.dsl.silu(gate_tile)
    result = ax.dsl.mul(activated, value_tile)

    ax.dsl.store(x=result, r=output[start:stop])
    return output
```

## 3. 数据流

```text
gate DDR slice ─→ OCM gate tile ─→ silu ─→ activated OCM
value DDR slice ─→ OCM value tile ───────────┘
                                      ↓
                                    mul
                                      ↓
                                result OCM
                                      ↓
                              output DDR slice
```

关键点：`activated` 没有显式 `store`，也没有被重新 `load`。

## 4. 与未融合版本对比

概念上的未融合流程：

```text
store activated → DDR
load activated ← DDR
mul
store result → DDR
```

融合后的机会：

```text
silu 与 mul 在局部 tile 中连接
减少中间 tensor 生命周期
减少 DDR traffic
减少 job 数
```

实际融合是否发生取决于后端模式匹配和 dtype / shape 支持，应通过 graph dump 与性能报告确认。

## 5. 启动与验证

```python
def fused_silu_mul(gate, value, output, tile_size=128):
    grid = (gate.shape[0] + tile_size - 1) // tile_size
    fused_silu_mul_kernel[grid](gate, value, output, tile_size)
```

运行：

```bash
python3 docs/examples/fused_silu_mul.py
```

## 6. 2D MLP 形式

实际 MLP 通常是二维：

```text
gate: (tokens, hidden)
value: (tokens, hidden)
output: (tokens, hidden)
```

可以按 token 行切分：

```python
pid = ax.program_id(0)
row0 = pid * TILE_TOKENS
row1 = min(row0 + TILE_TOKENS, tokens)

gate_tile = ax.dsl.load(gate[row0:row1, :])
value_tile = ax.dsl.load(value[row0:row1, :])
```

也可以进一步 tile hidden 维度，用于减小 OCM 工作集。

## 7. 与 MatMul 组合

常见 MLP：

```text
h = x @ w0
g = SiLU(h) * u
y = g @ w1
```

AXDSL 表达顺序：

```python
h = ax.dsl.matmul(x, w0, transposed=True)
u = ax.dsl.matmul(x, w1, transposed=True)
act = ax.dsl.silu(h)
g = ax.dsl.mul(act, u)
y = ax.dsl.matmul(g, w2, transposed=True)
```

优化方向：

- 权重常量
- tile 化 matmul
- activation 与 mul 融合
- 减少 `h` / `u` 落 DDR
- 分析 GEMM 与 elementwise 的 overlap

## 8. 精度注意

SiLU 涉及 `exp` 和除法：

```text
SiLU(x) = x / (1 + exp(-x))
```

bf16 / fp16 下可能出现：

- 小值精度损失
- 大值饱和差异
- reference 与 DSL 计算顺序差异

建议：

- fp32 先验证
- bf16 使用合理 tolerance
- 需要时研究中间 `output_dtype`
- 对部署数据分布做回归测试

## 9. 练习

### 练习 1：替换激活

分别尝试：

```python
ax.dsl.gelu
ax.dsl.quick_gelu
ax.dsl.sigmoid
ax.dsl.tanh
```

并同步修改 reference。

### 练习 2：添加 scale

实现：

```text
y = SiLU(gate) * value * scale
```

### 练习 3：二维 tile

将一维 tile 扩展为 `(TOKEN_TILE, HIDDEN_TILE)` 二维 grid。

### 练习 4：验证融合

开启 graph dump，查找 fused 前后图的算子数量与中间 tensor。

### 练习 5：性能对比

比较不同 tile 下的 cycles 和 DDR traffic。

## 10. 小结

这个例子说明 AXDSL 的融合优势不是靠隐式魔法，而是：

```text
局部 tile 数据 + 连续 DSL 算子 + 图级模式匹配
```

开发者把数据流写清楚，后端才有机会消除不必要的中间搬运。
