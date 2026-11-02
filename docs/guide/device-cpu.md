# DEVICE_CPU：设备 Arm CPU 后端

AXDSL 除 AX8860 NPU 后端外，还提供 `BACKEND.DEVICE_CPU`。该后端把 DSL 图导出为可在 **AXERA 设备端 Arm CPU** 上执行的 CPU runner，适合：

- 在无 NPU 编译条件时做小规模语义验证
- 为设备端 CPU pre-processing / post-processing 编写算子
- 复用同一套 AXDSL 表达方式连接 NPU 与 CPU 计算路径
- 为自定义算子提供 CPU fallback 或调试路径

::: important
`DEVICE_CPU` 不是指开发主机的 x86_64 CPU，而是 AXERA 设备上的 Arm CPU。在线运行时仍需要 AXCL、`torch_axera` 和设备运行环境。
:::

## 1. 选择后端

```python
from axnn.backend import BACKEND

import axnn.dsl as ax


@ax.builder(
    backend=BACKEND.DEVICE_CPU,
    run_mode=ax.RunMode.BRUN,
)
def cpu_add(x, y, output):
    ax.dsl.add(x, y, r=output)
    return output
```

关键点：

- `backend=BACKEND.DEVICE_CPU` 指定设备 CPU 后端
- 通常与 `RunMode.BRUN` 配合在线执行
- 输入输出是 AXERA device tensor
- launch 配置必须是单 chip、单 core、单 program

## 2. 设备 Arm CPU 在线运行

下面示例来自源码 `axnn/axnn/dsl/demo/brun_demo/brun_cpu.py`：

```python
import torch
import torch_axera  # noqa: F401  # Registers the AXERA device with PyTorch.

from axnn.backend import BACKEND
import axnn.dsl as ax


@ax.builder(backend=BACKEND.DEVICE_CPU)
def cpu_arithmetic_kernel(a, b, output):
    added = ax.dsl.add(a, b)
    subtracted = ax.dsl.sub(a, b)
    multiplied = ax.dsl.mul(added, subtracted)
    ax.dsl.div(multiplied, b, r=output)
    return output


ax.init(chip_num=1, core_num=1)
ax.init_session()

try:
    device = "axera:0"
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)

    a_device = a.to(device)
    b_device = b.to(device)
    output_device = torch.empty_like(a_device)

    cpu_arithmetic_kernel[1](a_device, b_device, output_device)
    ax.synchronize_stream()

    result = output_device.cpu()
finally:
    ax.finalize()
```

执行流程：

```text
Torch CPU tensor
  ↓ .to("axera:0")
AXERA device tensor
  ↓ AXDSL / DEVICE_CPU graph
Arm CPU runner
  ↓ ax.synchronize_stream()
Torch CPU tensor
```

## 3. 离线编译与 pulsar2 运行

`KernelBuilder.compile()` 可以把 `DEVICE_CPU` 图导出为 AXModel：

```python
from axnn.backend import BACKEND
import axnn.dsl as ax


@ax.builder(
    backend=BACKEND.DEVICE_CPU,
    enable_cache=False,
    cache_path="build/cache",
    dump_path="build/output",
)
def kernel(a, b, output):
    ax.dsl.add(a, b, r=output)
    return output


ax.init(chip_num=1, core_num=1)
model_path = kernel[1, 1].compile(a, b, output)
```

然后可使用 `pulsar2 run`：

```bash
mkdir -p input output

python3 - <<'PY'
import numpy as np
np.asarray([1, 2, 3, 4], dtype=np.float32).tofile("input/a.bin")
np.asarray([10, 20, 30, 40], dtype=np.float32).tofile("input/b.bin")
PY

pulsar2 run \
  --model "$MODEL_PATH" \
  --input_dir input \
  --output_dir output
```

输出文件名和输入输出约定与工具链版本相关；部署脚本应以后端生成的模型元数据为准。

## 4. 当前限制

当前源码中的 `DEVICE_CPU` 编译路径有以下明确约束：

| 约束 | 说明 |
|---|---|
| 单 chip | `chip_num` 必须为 1 |
| 单 core | `core_num` 必须为 1 |
| 单 program | launch grid 必须只有一个 program |
| 无 chip 级操作 | 不支持 `shmem_put` 等多芯片操作 |
| 完整 tensor 输入输出 | `load` / `store` 只支持完整 tensor，不支持 partial store |
| view 只读 | DEVICE_CPU tensor views 是只读物化，不能作为可写状态 |
| MatMul | 只支持 ordinary matrix multiplication |
| Cast | 浮点转整数必须显式 `output_rounding=1`，即 toward zero |

因此它不是 AX8860 多核 NPU 路径的替代品，而是设备端 CPU 执行路径。

## 5. 常见支持能力

根据当前源码测试，DEVICE_CPU 路径覆盖的典型能力包括：

- 算术：`add`、`sub`、`mul`、`div`
- 比较 / 选择：`maximum`、`minimum`、`where`
- 数学与激活：`exp`、`sqrt`、`reciprocal`、`sigmoid`、`tanh`、`softplus`
- Shape：`reshape`、`transpose`、`concat`、`expand`
- Reduction：`sum`、`max`、`min`
- `softmax`
- `matmul`
- `clip`
- `cast`，带 DEVICE_CPU 舍入约束
- `floor`，浮点输入、INT32 输出

完整能力以目标工具链的测试和 backend 支持矩阵为准。

## 6. 与 AX8860 NPU 路径的关系

| 维度 | AX8860 NPU | DEVICE_CPU |
|---|---|---|
| 计算单元 | AX8860 NPU core / engine | AXERA 设备 Arm CPU |
| 典型用途 | 高性能 Kernel | CPU fallback、前后处理、语义验证 |
| tile / OCM | 显式 NPU 内存层级 | 无 NPU OCM 编程模型 |
| 多核 / 多芯片 | AXDSL 支持相关表达 | 当前只支持单 chip、单 core、单 program |
| 在线执行 | AXCL / BRUN | AXCL / BRUN + CPU runner |
| 离线运行 | AXModel | 包含 CPU runner function 的 AXModel |

## 7. 建议

- 用 PYRUN 先确认算子语义
- 再用 DEVICE_CPU 验证设备端 CPU runner 可执行
- 最后切换到 `BACKEND.AX8860` 做 NPU 编译、调度与性能调优
- 不要把 DEVICE_CPU 的性能结论直接外推到 AX8860 NPU

一个常见开发顺序是：

```text
PyTorch reference
  ↓
PYRUN
  ↓
DEVICE_CPU BRUN / compile
  ↓
AX8860 CRUN
  ↓
AX8860 BRUN
  ↓
Performance tuning
```
