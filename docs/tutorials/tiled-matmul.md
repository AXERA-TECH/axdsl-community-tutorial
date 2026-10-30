# Tiled MatMul：二维切分

本教程实现：

```text
C = A @ B.T
```

其中：

```text
A: (M, K)
B: (N, K)
C: (M, N)
```

重点学习：

- 二维 grid
- `program_id(0)` / `program_id(1)`
- tile 数据搬运
- `matmul(transposed=True)`
- 输出区域 ownership

## 1. 为什么 tile 化

直接把整个矩阵送入算子最简单，但真实 Kernel 常常需要考虑：

- OCM 容量
- core 并行度
- 输入复用
- 尾块
- 后端 job 形态

Tiled MatMul 是学习 NPU Kernel 的经典案例。

## 2. Program 映射

每个 program 计算一个输出 tile：

```text
C[m0:m1, n0:n1]
```

需要的数据：

```text
A[m0:m1, :]        # (TM, K)
B[n0:n1, :]        # (TN, K)
```

二维 grid：

```text
grid = (ceil(M / TM), ceil(N / TN))
```

坐标：

```python
pid_m = ax.program_id(0)
pid_n = ax.program_id(1)
```

## 3. 输出 tile 表

以：

```text
M=256, N=256, TM=64, TN=64
```

为例：

```text
grid = (4, 4)
```

| `(pid_m, pid_n)` | A tile | B tile | C tile |
|---|---|---|---|
| `(0,0)` | `A[0:64, :]` | `B[0:64, :]` | `C[0:64, 0:64]` |
| `(0,1)` | `A[0:64, :]` | `B[64:128, :]` | `C[0:64, 64:128]` |
| ... | ... | ... | ... |
| `(3,3)` | `A[192:256, :]` | `B[192:256, :]` | `C[192:256, 192:256]` |

每个 C tile 只有一个 owner，避免写冲突。

## 4. Kernel 代码

```python
@ax.builder(run_mode=ax.RunMode.PYRUN)
def tiled_matmul_kernel(a, b, output, tile_size):
    tile_m, tile_n = tile_size

    pid_m = ax.program_id(0)
    pid_n = ax.program_id(1)

    m0 = pid_m * tile_m
    m1 = m0 + tile_m
    n0 = pid_n * tile_n
    n1 = n0 + tile_n

    a_tile = ax.dsl.load(a[m0:m1, :])
    b_tile = ax.dsl.load(b[n0:n1, :])

    c_tile = ax.dsl.matmul(a_tile, b_tile, transposed=True)
    ax.dsl.store(x=c_tile, r=output[m0:m1, n0:n1])
    return output
```

## 5. 启动

```python
def tiled_matmul(a, b, output, tile_m=128, tile_n=128):
    if a.shape[0] % tile_m or output.shape[1] % tile_n:
        raise ValueError("this tutorial requires M and N divisible by tile sizes")

    grid = (a.shape[0] // tile_m, output.shape[1] // tile_n)
    tiled_matmul_kernel[(grid, 1)](a, b, output, (tile_m, tile_n))
```

首版教程刻意要求 M / N 可整除，便于聚焦二维映射。生产 Kernel 需要处理尾块或 padding。

## 6. 验证

```python
reference = a @ b.transpose(0, 1)
tiled_matmul(a, b, output, tile_m=64, tile_n=64)
torch.testing.assert_close(reference, output, atol=2e-2, rtol=2e-2)
```

bf16 使用较宽的 tolerance。

运行：

```bash
python3 docs/examples/tiled_matmul.py
```

## 7. 数据流

每个 program：

```text
A tile DDR ─┐
            ├─ load → OCM
B tile DDR ─┘
            ↓
      matmul(A_tile, B_tile.T)
            ↓
        C tile OCM
            ↓
      store to C DDR slice
```

## 8. tile 复用直觉

同一个 `A[m0:m1, :]` 会被 `N / TN` 个 program 使用：

```text
(pid_m=0, pid_n=0..N/TN-1)
```

同一个 `B[n0:n1, :]` 会被 `M / TM` 个 program 使用：

```text
(pid_m=0..M/TM-1, pid_n=0)
```

因此 tile 越大，表面复用越强，但 OCM 压力也越大。实际最优值应由 build context 与 trace 决定。

## 9. 支持尾块的思路

如果 `M` 或 `N` 不整除，需要：

```python
m1 = min(m0 + tile_m, M)
n1 = min(n0 + tile_n, N)
```

但尾块 tile shape 会变化，某些算子 / 后端组合要求 tile shape 固定。可选策略：

1. 要求输入 padding 到固定 tile
2. 为尾块生成独立路径
3. 使用多个 shape-specific kernel
4. 使用支持的 dynamic read/write API
5. 在框架层按 shape group 编译不同版本

社区示例建议先明确当前工具链支持哪种方式，再决定实现。

## 10. 练习

### 练习 1：修改 tile size

比较：

```text
(32, 64)
(64, 64)
(64, 128)
(128, 64)
```

记录 grid 数量与程序输出。

### 练习 2：改成 CRUN

在可用工具链下改为 CRUN，并开启：

```python
check=ax.Check.ALL_CLOSE
check_mode=ax.CheckMode.CHECK_PER_LAYER
enable_cache=False
```

### 练习 3：融合 bias

实现：

```text
C = A @ B.T + bias
```

可先用 elementwise `add`，再研究 `matmul` 的常量权重 / bias 参数。

### 练习 4：多核显式指定

尝试：

```python
kernel[(grid_m, grid_n), core_num](...)
```

观察 core 数限制和调度行为。

### 练习 5：性能报告

开启 `dump_dsl_perf`，比较不同 tile 的：

- cycles
- DDR traffic
- OCM usage
- jobs
- utilization

## 11. 下一步

进入 [Fused SiLU × Mul](/tutorials/fused-silu-mul)，学习算子融合如何减少中间数据。
