# 完整算子与 API 列表

本页按能力列出当前源码公开导出的 AXDSL API。签名与说明直接来自 `axnn/axnn/dsl` 源码 docstring，便于与工具链版本对齐。教程中常用的语义解释见 [常用算子与 PyTorch 迁移](/advanced/operators)。

## 快速能力索引

| 类别 | API |
|---|---|
| Kernel 与上下文 | [`ax.builder`](#axbuilder)<br>[`ax.init`](#axinit)<br>[`ax.program_id`](#axprogram_id)<br>[`ax.num_programs`](#axnum_programs)<br>[`ax.init_session`](#axinit_session)<br>[`ax.synchronize_stream`](#axsynchronize_stream)<br>[`ax.finalize`](#axfinalize) |
| 内存与数据搬运 | [`ax.dsl.load`](#axdslload)<br>[`ax.dsl.store`](#axdslstore)<br>[`ax.dsl.move`](#axdslmove)<br>[`ax.dsl.dynamic_ld`](#axdsldynamic_ld)<br>[`ax.dsl.dynamic_st`](#axdsldynamic_st) |
| Tensor 创建与常量 | [`ax.dsl.create_tensor`](#axdslcreate_tensor)<br>[`ax.dsl.zeros`](#axdslzeros)<br>[`ax.dsl.full`](#axdslfull)<br>[`ax.dsl.from_numpy`](#axdslfrom_numpy)<br>[`ax.dsl.cast`](#axdslcast) |
| Elementwise 与数学 | [`ax.dsl.add`](#axdsladd)<br>[`ax.dsl.sub`](#axdslsub)<br>[`ax.dsl.mul`](#axdslmul)<br>[`ax.dsl.div`](#axdsldiv)<br>[`ax.dsl.maximum`](#axdslmaximum)<br>[`ax.dsl.minimum`](#axdslminimum)<br>[`ax.dsl.clip`](#axdslclip)<br>[`ax.dsl.where`](#axdslwhere)<br>[`ax.dsl.exp`](#axdslexp)<br>[`ax.dsl.log2`](#axdsllog2)<br>[`ax.dsl.sqrt`](#axdslsqrt)<br>[`ax.dsl.reciprocal`](#axdslreciprocal)<br>[`ax.dsl.floor`](#axdslfloor)<br>[`ax.dsl.ceil`](#axdslceil) |
| 激活函数 | [`ax.dsl.sigmoid`](#axdslsigmoid)<br>[`ax.dsl.tanh`](#axdsltanh)<br>[`ax.dsl.gelu`](#axdslgelu)<br>[`ax.dsl.quick_gelu`](#axdslquick_gelu)<br>[`ax.dsl.silu`](#axdslsilu)<br>[`ax.dsl.softplus`](#axdslsoftplus) |
| Reduction 与搜索 | [`ax.dsl.sum`](#axdslsum)<br>[`ax.dsl.max`](#axdslmax)<br>[`ax.dsl.min`](#axdslmin)<br>[`ax.dsl.argmax`](#axdslargmax)<br>[`ax.dsl.argmin`](#axdslargmin)<br>[`ax.dsl.cumsum`](#axdslcumsum)<br>[`ax.dsl.softmax`](#axdslsoftmax)<br>[`ax.dsl.topk`](#axdsltopk)<br>[`ax.dsl.nonzero`](#axdslnonzero) |
| GEMM 与卷积 | [`ax.dsl.matmul`](#axdslmatmul)<br>[`ax.dsl.conv`](#axdslconv) |
| Shape 与索引 | [`ax.dsl.reshape`](#axdslreshape)<br>[`ax.dsl.transpose`](#axdsltranspose)<br>[`ax.dsl.concat`](#axdslconcat)<br>[`ax.dsl.expand`](#axdslexpand)<br>[`ax.dsl.pad`](#axdslpad)<br>[`ax.dsl.gather`](#axdslgather)<br>[`ax.dsl.scatter`](#axdslscatter)<br>[`ax.dsl.index_select`](#axdslindex_select)<br>[`ax.dsl.index_add`](#axdslindex_add)<br>[`ax.dsl.select_scatter`](#axdslselect_scatter) |
| 结构化控制流 | [`ax.dsl.if_op`](#axdslif_op)<br>[`ax.dsl.while_loop`](#axdslwhile_loop)<br>[`ax.dsl.range`](#axdslrange)<br>[`ax.dsl.switch`](#axdslswitch) |
| 调试 | [`ax.dsl.device_print`](#axdsldevice_print) |
| 多核与多芯片并行 | [`ax.parallel.sync_all`](#axparallelsync_all)<br>[`ax.parallel.shmem_create_tensor`](#axparallelshmem_create_tensor)<br>[`ax.parallel.shmem_create_signal`](#axparallelshmem_create_signal)<br>[`ax.parallel.shmem_put`](#axparallelshmem_put)<br>[`ax.parallel.shmem_put_signal_nbi`](#axparallelshmem_put_signal_nbi)<br>[`ax.parallel.shmem_signal_update`](#axparallelshmem_signal_update)<br>[`ax.parallel.signal_update`](#axparallelsignal_update)<br>[`ax.parallel.signal_wait`](#axparallelsignal_wait)<br>[`ax.parallel.pub`](#axparallelpub)<br>[`ax.parallel.sub`](#axparallelsub)<br>[`ax.host.shmem_create_tensor`](#axhostshmem_create_tensor)<br>[`ax.host.shmem_create_signal`](#axhostshmem_create_signal) |

## 枚举与 dtype

### RunMode

| 值 | 说明 |
|---|---|
| `PYRUN` | 使用 Python reference implementation |
| `CRUN` | 编译后使用 backend simulator |
| `BRUN` | 编译 AXModel 后在物理设备 / AXCL runtime 执行 |

### Check

| 值 | 说明 |
|---|---|
| `NO_CHECK` | 不比较 PYRUN 与 CRUN |
| `ALL_CLOSE` | 在 rtol / atol 内接近 |
| `ALL_EQUAL` | 精确相等 |
| `COSINE_SIMILARITY` | 余弦相似度达到阈值 |

### CheckMode

| 值 | 说明 |
|---|---|
| `CHECK_OUTPUT` | 只检查最终输出 |
| `CHECK_PER_LAYER` | 检查中间层结果 |

### MemoryType

| 值 | 说明 |
|---|---|
| `DDR` | chip 对应外部内存，适合公开输入输出与大 buffer |
| `OCM` | core 私有片上内存，适合局部 tile 与临时变量 |
| `SHARED_OCM` | chip 内 core 共享片上内存 |
| `SIGNAL` | 同步信号地址空间 |

### SigOp

| 值 | 说明 |
|---|---|
| `SET` | 将 signal 设置为指定值 |
| `ADD` | 将指定值累加到 signal |

### dtype

| 类别 | 名称 |
|---|---|
| 整数 | `ax.int32`, `ax.uint32`, `ax.int16`, `ax.uint16`, `ax.int8`, `ax.uint8`, `ax.int4`, `ax.uint4` |
| 浮点 | `ax.fp32`, `ax.fp16`, `ax.bf16`, `ax.fp8_e4m3`, `ax.fp8_e5m2` |
| 微缩放 / 块缩放 | `ax.mxfp8_e4m3`, `ax.mxfp4`, `ax.mxe8m0`, `ax.nvfp4` |

::: warning
dtype 的算子级支持情况不仅取决于 AXDSL 类型定义，还取决于目标 backend 与编译器版本。发布社区文档时，建议补充每个算子的 dtype / shape 支持矩阵。
:::

## API 详细列表

## Kernel 与上下文
<h3 id="axbuilder">ax.builder</h3>

**签名**

```python
def builder(
    _func=None,
    /,
    *,
    const_inputs=None,
    check=Check.NO_CHECK,
    check_mode=CheckMode.CHECK_OUTPUT,
    check_rtol=1e-05,
    check_atol=0,
    check_cosine_similarity=0.999,
    check_tensor_black_list=None,
    run_mode=RunMode.BRUN,
    backend=None,
    enable_cache=True,
    unique_axmodel=False,
    cache_path=None,
    dump_dsl_perf=False,
    dump_path=None,
)

```

**功能概述**

Decorate a Python function as a launchable AXDSL kernel.

The returned builder records the function into AXDSL graph IR when it is launched with a grid, for example `kernel[4](...)`. Execution can use the Python reference implementation or specific backend. Correctness checking and compilation caching can be configured in decorator params.

| 参数 | 说明 |
|---|---|
| `_func` | Function supplied as the kernel. Users normally omit this argument. |
| `const_inputs` | Kernel parameter names treated as compile-time Python or NumPy constants. |
| `check` | Output comparison policy used by CRUN against a PYRUN reference. |
| `check_mode` | Compare final outputs or recorded intermediate outputs. |
| `check_rtol` | Relative tolerance for `Check.ALL_CLOSE`. |
| `check_atol` | Absolute tolerance for `Check.ALL_CLOSE`. |
| `check_cosine_similarity` | Minimum similarity for `Check.COSINE_SIMILARITY`. |
| `check_tensor_black_list` | Optional tensor-name patterns excluded from per-layer checks. |
| `run_mode` | Execute with the Python reference implementation or specific backend. |
| `backend` | `BACKEND.AX8860` or `BACKEND.DEVICE_CPU`. `None` selects AX8860. |
| `enable_cache` | Reuse compilation artifacts. Cache and correctness checking cannot be enabled together. |
| `unique_axmodel` | Reuse one loaded AXModel for every BRUN launch of this kernel. |
| `cache_path` | Directory for compilation cache entries. Defaults to `~/.cache/axdsl`. |
| `dump_dsl_perf` | Dump the kernel source, raw and fused graphs, compiled job metadata, and simulator cycle trace for offline performance analysis. This requires a cache-disabled, CRUN launch. After simulation, the builder logs the command that generates the JSON and HTML reports. |
| `dump_path` | Directory for compiled models and debug artifacts. Defaults to the `output` directory under the current working directory. |

**Returns**

- A decorator, or a launchable kernel for the no-parentheses form.

**Raises**

- AssertionError: If correctness checking and compilation caching are both enabled.
- ValueError: If DSL performance artifacts are requested for a cached, non-CRUN, or multi-chip launch.

<h3 id="axinit">ax.init</h3>

**签名**

```python
def init(
    chip_num=1,
    core_num=1,
)

```

**功能概述**

Initialize the AXDSL recording context.

Call this before launching a kernel. Reinitializing resets the chip configuration.

| 参数 | 说明 |
|---|---|
| `chip_num` | Number of chips participating in the launch. |
| `core_num` | Accepted for compatibility. Core count is selected by each kernel launch. |

**Returns**

- None.

<h3 id="axprogram_id">ax.program_id</h3>

**签名**

```python
def program_id(
    axis,
)

```

**功能概述**

Return the current program coordinate on one launch-grid axis.

| 参数 | 说明 |
|---|---|
| `axis` | Zero-based index of launch-grid axis. |

**Returns**

- Current integer coordinate on `axis`.

<h3 id="axnum_programs">ax.num_programs</h3>

**签名**

```python
def num_programs(
    axis,
)

```

**功能概述**

Return the number of programs on one launch-grid axis.

| 参数 | 说明 |
|---|---|
| `axis` | Zero-based index of launch-grid axis. |

**Returns**

- Integer grid extent on `axis`.

<h3 id="axinit_session">ax.init_session</h3>

**签名**

```python
def init_session()

```

**功能概述**

Initialize resources shared by every BRUN kernel on the selected device.

<h3 id="axsynchronize_stream">ax.synchronize_stream</h3>

**签名**

```python
def synchronize_stream()

```

**功能概述**

Wait for work submitted on the most recently selected AXCL stream.

<h3 id="axfinalize">ax.finalize</h3>

**签名**

```python
def finalize()

```

**功能概述**

Release loaded BRUN models, then shared AXCL resources.


## 内存与数据搬运
<h3 id="axdslload">ax.dsl.load</h3>

**签名**

```python
def load(
    x: Tensor,
    *,
    r: Optional[Tensor]=None,
) -> Tensor

```

**功能概述**

Loads a DDR tensor into non-DDR memory.

| 参数 | 说明 |
|---|---|
| `x` | DDR input tensor. |
| `r` | Optional non-DDR output tensor. |

**Returns**

- Tensor loaded into non-DDR memory.

<h3 id="axdslstore">ax.dsl.store</h3>

**签名**

```python
def store(
    x: Tensor,
    *,
    r: Tensor,
) -> None

```

**功能概述**

Stores a non-DDR tensor into a DDR target.

| 参数 | 说明 |
|---|---|
| `x` | Non-DDR source tensor. |
| `r` | DDR target tensor. |

**Returns**

- None.

<h3 id="axdslmove">ax.dsl.move</h3>

**签名**

```python
def move(
    x: Tensor,
    *,
    r: Tensor,
    op: Optional[str]=None,
) -> None

```

**功能概述**

Copies tensor data between compatible memory regions.

The source and destination tensors must both be non-DDR tensors or both be DDR tensors.

| 参数 | 说明 |
|---|---|
| `x` | Source tensor. |
| `r` | Destination tensor. |
| `op` | Optional move mode. If provided, only `"add"` is supported. |

**Returns**

- None.

<h3 id="axdsldynamic_ld">ax.dsl.dynamic_ld</h3>

**签名**

```python
def dynamic_ld(
    x: Tensor,
    src_offset: Tensor,
    shape: Tensor,
    *,
    dim: tuple[int, ...],
    output_shape: Optional[tuple[int, ...]]=None,
    r: Optional[Tensor]=None,
) -> Tensor

```

**功能概述**

Load a runtime-selected region from DDR into OCM.

The operation uses `src_offset` and `shape` to select a region along the specified dimensions in `dim`.

| 参数 | 说明 |
|---|---|
| `x` | DDR source tensor. |
| `src_offset` | Runtime starting indices for the dimensions in `dim`. |
| `shape` | Runtime region sizes for the dimensions in `dim`. |
| `dim` | Source dimensions selected by `src_offset` and `shape`. |
| `output_shape` | Static shape of the OCM result buffer. Required when `r` is omitted. |
| `r` | Optional caller-provided OCM output tensor. |

**Returns**

- The loaded OCM tensor.

**Raises**

- AssertionError: If `x` is not in DDR, `r` is in DDR, or neither `r` nor `output_shape` is provided.

<h3 id="axdsldynamic_st">ax.dsl.dynamic_st</h3>

**签名**

```python
def dynamic_st(
    x: Tensor,
    src_offset: Tensor,
    dst_offset: Tensor,
    shape: Tensor,
    *,
    dim: tuple[int, ...],
    output_shape: Optional[tuple[int, ...]]=None,
    r: Optional[Tensor]=None,
)

```

**功能概述**

Store a runtime-selected region from OCM into DDR.

The operation uses `src_offset`, `dst_offset`, and `shape` to select a region along the specified dimensions in `dim`. The selected region is copied from `x` to the corresponding destination region in the DDR output tensor.

| 参数 | 说明 |
|---|---|
| `x` | OCM source tensor. |
| `src_offset` | Runtime starting indices for the source dimensions in `dim`. |
| `dst_offset` | Runtime starting indices for the destination dimensions in `dim`. |
| `shape` | Runtime region sizes for the dimensions in `dim`. |
| `dim` | Source and destination dimensions selected by the offsets and `shape`. |
| `output_shape` | Static shape of the DDR result buffer. Required when `r` is omitted. |
| `r` | Optional caller-provided DDR output tensor. If omitted, a DDR tensor is created. |

**Returns**

- The stored DDR tensor.

**Raises**

- AssertionError: If `x` is in DDR, `r` is not in DDR, or neither `r` nor `output_shape` is provided.


## Tensor 创建与常量
<h3 id="axdslcreate_tensor">ax.dsl.create_tensor</h3>

**签名**

```python
def create_tensor(
    shape: Tuple[int, ...],
    dtype: Union[DslDtype, DType],
    memory_type: Optional[MemoryType]=MemoryType.OCM,
    name: Optional[str]=None,
) -> Tensor

```

**功能概述**

Creates and registers an AXDSL tensor.

| 参数 | 说明 |
|---|---|
| `shape` | Tensor shape. |
| `dtype` | Tensor dtype. |
| `memory_type` | Target memory type. |
| `name` | Optional tensor name. |

**Returns**

- Registered AXDSL tensor metadata.

<h3 id="axdslzeros">ax.dsl.zeros</h3>

**签名**

```python
def zeros(
    shape: Tuple[int, ...],
    dtype: Union[DslDtype, DType],
    memory_type: Optional[MemoryType]=MemoryType.OCM,
    prefix: Optional[str]='zeros',
) -> Tensor

```

**功能概述**

Creates a tensor filled with zeros.

| 参数 | 说明 |
|---|---|
| `shape` | Tensor shape. |
| `dtype` | Tensor dtype. |
| `memory_type` | Target memory type. |
| `prefix` | Optional name prefix used for generated tensors. |

**Returns**

- Zero-filled AXDSL tensor.

<h3 id="axdslfull">ax.dsl.full</h3>

**签名**

```python
def full(
    shape: Tuple[int, ...],
    dtype: Union[DslDtype, DType],
    fill_value: Union[int, float],
    memory_type: Optional[MemoryType]=MemoryType.OCM,
    prefix: Optional[str]='full',
) -> Tensor

```

**功能概述**

Creates a tensor filled with a scalar value.

| 参数 | 说明 |
|---|---|
| `shape` | Tensor shape. |
| `dtype` | Tensor dtype. |
| `fill_value` | Scalar fill value. |
| `memory_type` | Target memory type. |
| `prefix` | Optional name prefix used for generated tensors. |

**Returns**

- Scalar-filled AXDSL tensor.

<h3 id="axdslfrom_numpy">ax.dsl.from_numpy</h3>

**签名**

```python
def from_numpy(
    data: np.ndarray,
    prefix: Optional[str]='from_numpy',
    *,
    r: Optional[Tensor]=None,
) -> Tensor

```

**功能概述**

Registers a NumPy array as an AXDSL constant tensor and loads it to on-chip memory.

| 参数 | 说明 |
|---|---|
| `data` | NumPy array to register. |
| `prefix` | Constant tensor name prefix. |
| `r` | Optional non-DDR destination tensor. |

**Returns**

- On-chip tensor loaded from the registered constant.

<h3 id="axdslcast">ax.dsl.cast</h3>

**签名**

```python
def cast(
    x: Tensor,
    *,
    dtype: Union[DslDtype, DType],
    output_rounding: Optional[int]=None,
    r: Optional[Tensor]=None,
) -> Tensor

```

**功能概述**

Converts a tensor to a target dtype.

| 参数 | 说明 |
|---|---|
| `x` | Input tensor. |
| `dtype` | Target dtype. |
| `output_rounding` | Optional float-to-integer rounding mode. 0 is nearest-even, 1 is toward-zero, 2 is toward-positive-infinity, 3 is toward-negative-infinity, 4 is nearest-up, and 5 is away-from-zero. |
| `r` | Optional output tensor. |

**Returns**

- Tensor converted to the target dtype.


## Elementwise 与数学
<h3 id="axdsladd">ax.dsl.add</h3>

**签名**

```python
def add(
    x: Tensor,
    y: Tensor,
    *,
    output_dtype: typing_dtype=None,
    r: Optional[Tensor]=None,
) -> Tensor

```

**功能概述**

Adds two tensors elementwise.

| 参数 | 说明 |
|---|---|
| `x` | Left input tensor. |
| `y` | Right input tensor. |
| `output_dtype` | Optional output dtype. |
| `r` | Optional output tensor. |

**Returns**

- Elementwise sum tensor.

<h3 id="axdslsub">ax.dsl.sub</h3>

**签名**

```python
def sub(
    x: Tensor,
    y: Tensor,
    *,
    output_dtype: typing_dtype=None,
    r: Optional[Tensor]=None,
) -> Tensor

```

**功能概述**

Subtracts y from x elementwise.

| 参数 | 说明 |
|---|---|
| `x` | Left input tensor. |
| `y` | Right input tensor. |
| `output_dtype` | Optional output dtype. |
| `r` | Optional output tensor. |

**Returns**

- Elementwise difference tensor.

<h3 id="axdslmul">ax.dsl.mul</h3>

**签名**

```python
def mul(
    x: Tensor,
    y: Tensor,
    *,
    output_dtype: typing_dtype=None,
    r: Optional[Tensor]=None,
) -> Tensor

```

**功能概述**

Multiplies two tensors elementwise.

| 参数 | 说明 |
|---|---|
| `x` | Left input tensor. |
| `y` | Right input tensor. |
| `output_dtype` | Optional output dtype. |
| `r` | Optional output tensor. |

**Returns**

- Elementwise product tensor.

<h3 id="axdsldiv">ax.dsl.div</h3>

**签名**

```python
def div(
    x: Tensor,
    y: Tensor,
    *,
    output_dtype: typing_dtype=None,
    output_rounding: int=None,
    high_precision: Optional[bool]=False,
    r: Optional[Tensor]=None,
) -> Tensor

```

**功能概述**

Divides x by y elementwise.

| 参数 | 说明 |
|---|---|
| `x` | Numerator tensor. |
| `y` | Denominator tensor. |
| `output_dtype` | Optional output dtype. |
| `output_rounding` | Optional float-to-integer rounding mode. |
| `high_precision` | Optional flag to use the hardware divider instead of reciprocal multiplication. |
| `r` | Optional output tensor. |

**Returns**

- Elementwise quotient tensor.

<h3 id="axdslmaximum">ax.dsl.maximum</h3>

**签名**

```python
def maximum(
    x: Tensor,
    y: Tensor,
    *,
    output_dtype: typing_dtype=None,
    r: Optional[Tensor]=None,
) -> Tensor

```

**功能概述**

Computes the elementwise maximum of two tensors.

| 参数 | 说明 |
|---|---|
| `x` | Left input tensor. |
| `y` | Right input tensor. |
| `output_dtype` | Optional output dtype. |
| `r` | Optional output tensor. |

**Returns**

- Elementwise maximum tensor.

<h3 id="axdslminimum">ax.dsl.minimum</h3>

**签名**

```python
def minimum(
    x: Tensor,
    y: Tensor,
    *,
    output_dtype: typing_dtype=None,
    r: Optional[Tensor]=None,
) -> Tensor

```

**功能概述**

Computes the elementwise minimum of two tensors.

| 参数 | 说明 |
|---|---|
| `x` | Left input tensor. |
| `y` | Right input tensor. |
| `output_dtype` | Optional output dtype. |
| `r` | Optional output tensor. |

**Returns**

- Elementwise minimum tensor.

<h3 id="axdslclip">ax.dsl.clip</h3>

**签名**

```python
def clip(
    x: Tensor,
    *,
    min: Union[int, float],
    max: Union[int, float],
    r: Optional[Tensor]=None,
) -> Tensor

```

**功能概述**

Clamps tensor values into a closed interval.

$$r_i = \min\!\left(\max(x_i, \mathrm{min}), \mathrm{max}\right)$$

| 参数 | 说明 |
|---|---|
| `x` | Input tensor. |
| `min` | Lower bound. |
| `max` | Upper bound. |
| `r` | Optional output tensor. |

**Returns**

- Clipped tensor.

<h3 id="axdslwhere">ax.dsl.where</h3>

**签名**

```python
def where(
    cond: Tensor,
    x: Tensor,
    y: Tensor,
    *,
    output_dtype: typing_dtype=None,
    r: Optional[Tensor]=None,
) -> Tensor

```

**功能概述**

Selects values from x or y according to cond elementwise.

Equivalent to `torch.where`.

| 参数 | 说明 |
|---|---|
| `cond` | Condition tensor. |
| `x` | Tensor selected where cond is true. |
| `y` | Tensor selected where cond is false. |
| `output_dtype` | Optional output dtype. |
| `r` | Optional output tensor. |

**Returns**

- Condition-selected tensor.

<h3 id="axdslexp">ax.dsl.exp</h3>

**签名**

```python
def exp(
    x: Tensor,
    *,
    output_dtype: typing_dtype=None,
    r: Optional[Tensor]=None,
) -> Tensor

```

**功能概述**

Computes the elementwise natural exponential.

$$r_i = e^{x_i}$$

| 参数 | 说明 |
|---|---|
| `x` | Input tensor. |
| `output_dtype` | Optional output dtype. |
| `r` | Optional output tensor. |

**Returns**

- Exponential result tensor.

<h3 id="axdsllog2">ax.dsl.log2</h3>

**签名**

```python
def log2(
    x: Tensor,
    *,
    output_dtype: typing_dtype=None,
    r: Optional[Tensor]=None,
) -> Tensor

```

**功能概述**

Computes the elementwise base-2 logarithm.

$$r_i = \log_2(x_i)$$

| 参数 | 说明 |
|---|---|
| `x` | Input tensor. |
| `output_dtype` | Optional output dtype. |
| `r` | Optional output tensor. |

**Returns**

- Base-2 logarithm result tensor.

<h3 id="axdslsqrt">ax.dsl.sqrt</h3>

**签名**

```python
def sqrt(
    x: Tensor,
    *,
    output_dtype: typing_dtype=None,
    r: Optional[Tensor]=None,
) -> Tensor

```

**功能概述**

Computes the elementwise square root.

$$r_i = \sqrt{x_i}$$

| 参数 | 说明 |
|---|---|
| `x` | Input tensor. |
| `output_dtype` | Optional output dtype. |
| `r` | Optional output tensor. |

**Returns**

- Square-root result tensor.

<h3 id="axdslreciprocal">ax.dsl.reciprocal</h3>

**签名**

```python
def reciprocal(
    x: Tensor,
    *,
    output_dtype: typing_dtype=None,
    r: Optional[Tensor]=None,
) -> Tensor

```

**功能概述**

Computes the elementwise reciprocal.

| 参数 | 说明 |
|---|---|
| `x` | Input tensor. |
| `output_dtype` | Optional output dtype. |
| `r` | Optional output tensor. |

**Returns**

- Reciprocal result tensor.

<h3 id="axdslfloor">ax.dsl.floor</h3>

**签名**

```python
def floor(
    x: Tensor,
    *,
    r: Optional[Tensor]=None,
) -> Tensor

```

**功能概述**

Rounds each element toward negative infinity.

| 参数 | 说明 |
|---|---|
| `x` | Input tensor. |
| `r` | Optional output tensor. |

**Returns**

- Floor result tensor.

<h3 id="axdslceil">ax.dsl.ceil</h3>

**签名**

```python
def ceil(
    x: Tensor,
    *,
    r: Optional[Tensor]=None,
) -> Tensor

```

**功能概述**

Rounds each element toward positive infinity.

| 参数 | 说明 |
|---|---|
| `x` | Input tensor. |
| `r` | Optional output tensor. |

**Returns**

- Ceil result tensor.


## 激活函数
<h3 id="axdslsigmoid">ax.dsl.sigmoid</h3>

**签名**

```python
def sigmoid(
    x: Tensor,
    *,
    output_dtype: typing_dtype=None,
    r: Optional[Tensor]=None,
) -> Tensor

```

**功能概述**

Computes the elementwise sigmoid activation.

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

| 参数 | 说明 |
|---|---|
| `x` | Input tensor. |
| `output_dtype` | Optional output dtype. |
| `r` | Optional output tensor. |

**Returns**

- Sigmoid result tensor.

<h3 id="axdsltanh">ax.dsl.tanh</h3>

**签名**

```python
def tanh(
    x: Tensor,
    *,
    output_dtype: typing_dtype=None,
    r: Optional[Tensor]=None,
) -> Tensor

```

**功能概述**

Computes the elementwise hyperbolic tangent.

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

| 参数 | 说明 |
|---|---|
| `x` | Input tensor. |
| `output_dtype` | Optional output dtype. |
| `r` | Optional output tensor. |

**Returns**

- Tanh result tensor.

<h3 id="axdslgelu">ax.dsl.gelu</h3>

**签名**

```python
def gelu(
    x: Tensor,
    *,
    output_dtype: typing_dtype=None,
    r: Optional[Tensor]=None,
) -> Tensor

```

**功能概述**

Computes the GELU activation elementwise.

The exact GELU function is defined as:

$$\operatorname{GELU}(x) = \frac{x}{2} \left(1 + \operatorname{erf}\left(\frac{x}{\sqrt{2}}\right)\right)$$

Equivalent to `torch.nn.functional.gelu`.

| 参数 | 说明 |
|---|---|
| `x` | Input tensor. |
| `output_dtype` | Optional output dtype. |
| `r` | Optional output tensor. |

**Returns**

- GELU result tensor.

<h3 id="axdslquick_gelu">ax.dsl.quick_gelu</h3>

**签名**

```python
def quick_gelu(
    x: Tensor,
    *,
    output_dtype: typing_dtype=None,
    r: Optional[Tensor]=None,
) -> Tensor

```

**功能概述**

Computes the QuickGELU activation elementwise.

$$\operatorname{QuickGELU}(x) = x\,\sigma(1.702x)$$

where $\sigma$ is the sigmoid function.

| 参数 | 说明 |
|---|---|
| `x` | Input tensor. |
| `output_dtype` | Optional output dtype. |
| `r` | Optional output tensor. |

**Returns**

- QuickGELU result tensor.

<h3 id="axdslsilu">ax.dsl.silu</h3>

**签名**

```python
def silu(
    x: Tensor,
    *,
    output_dtype: typing_dtype=None,
    r: Optional[Tensor]=None,
) -> Tensor

```

**功能概述**

Computes the SiLU activation elementwise.

$$\operatorname{SiLU}(x) = x\,\sigma(x) = \frac{x}{1 + e^{-x}}$$

| 参数 | 说明 |
|---|---|
| `x` | Input tensor. |
| `output_dtype` | Optional output dtype. |
| `r` | Optional output tensor. |

**Returns**

- SiLU result tensor.

<h3 id="axdslsoftplus">ax.dsl.softplus</h3>

**签名**

```python
def softplus(
    x: Tensor,
    *,
    output_dtype: typing_dtype=None,
    r: Optional[Tensor]=None,
) -> Tensor

```

**功能概述**

Computes the Softplus activation elementwise.

$$\operatorname{Softplus}(x) = \log(1 + e^x)$$

Equivalent to `torch.nn.functional.softplus` with `beta=1` and `threshold=20`. AXDSL does not expose `beta` or `threshold` as configurable parameters.

| 参数 | 说明 |
|---|---|
| `x` | Input tensor. |
| `output_dtype` | Optional output dtype. |
| `r` | Optional output tensor. |

**Returns**

- Softplus result tensor.


## Reduction 与搜索
<h3 id="axdslsum">ax.dsl.sum</h3>

**签名**

```python
def sum(
    x: Tensor,
    *,
    dim: Union[int, List[int]],
    keepdim: Optional[bool]=False,
    output_dtype: typing_dtype=None,
    r: Optional[Tensor]=None,
) -> Tensor

```

**功能概述**

Sums tensor values over one or more dimensions.

| 参数 | 说明 |
|---|---|
| `x` | Input tensor. |
| `dim` | Reduction dimension or dimensions. |
| `keepdim` | Whether to keep reduced dimensions. |
| `output_dtype` | Optional output dtype. |
| `r` | Optional output tensor. |

**Returns**

- Sum-reduction result tensor.

<h3 id="axdslmax">ax.dsl.max</h3>

**签名**

```python
def max(
    x: Tensor,
    *,
    dim: Union[int, List[int]],
    keepdim: Optional[bool]=False,
    r: Optional[Tensor]=None,
) -> Tensor

```

**功能概述**

Takes the maximum over one or more dimensions.

| 参数 | 说明 |
|---|---|
| `x` | Input tensor. |
| `dim` | Reduction dimension or dimensions. |
| `keepdim` | Whether to keep reduced dimensions. |
| `r` | Optional output tensor. |

**Returns**

- Maximum-reduction result tensor.

<h3 id="axdslmin">ax.dsl.min</h3>

**签名**

```python
def min(
    x: Tensor,
    *,
    dim: Union[int, List[int]],
    keepdim: Optional[bool]=False,
    r: Optional[Tensor]=None,
) -> Tensor

```

**功能概述**

Takes the minimum over one or more dimensions.

| 参数 | 说明 |
|---|---|
| `x` | Input tensor. |
| `dim` | Reduction dimension or dimensions. |
| `keepdim` | Whether to keep reduced dimensions. |
| `r` | Optional output tensor. |

**Returns**

- Minimum-reduction result tensor.

<h3 id="axdslargmax">ax.dsl.argmax</h3>

**签名**

```python
def argmax(
    x: Tensor,
    *,
    dim: int,
    keepdim: Optional[bool]=True,
    r: Optional[Tensor]=None,
) -> Tensor

```

**功能概述**

Returns indices of maximum values along a dimension.

| 参数 | 说明 |
|---|---|
| `x` | Input tensor. |
| `dim` | Reduction dimension. |
| `keepdim` | Whether to keep the reduced dimension. |
| `r` | Optional output tensor. |

**Returns**

- Maximum-value index tensor.

<h3 id="axdslargmin">ax.dsl.argmin</h3>

**签名**

```python
def argmin(
    x: Tensor,
    *,
    dim: int,
    keepdim: Optional[bool]=True,
    r: Optional[Tensor]=None,
) -> Tensor

```

**功能概述**

Returns indices of minimum values along a dimension.

| 参数 | 说明 |
|---|---|
| `x` | Input tensor. |
| `dim` | Reduction dimension. |
| `keepdim` | Whether to keep the reduced dimension. |
| `r` | Optional output tensor. |

**Returns**

- Minimum-value index tensor.

<h3 id="axdslcumsum">ax.dsl.cumsum</h3>

**签名**

```python
def cumsum(
    x: Tensor,
    *,
    dim: int,
    output_dtype: typing_dtype=None,
    r: Optional[Tensor]=None,
) -> Tensor

```

**功能概述**

Computes an inclusive cumulative sum along one dimension.

Equivalent to `torch.cumsum` for forward, inclusive accumulation.

| 参数 | 说明 |
|---|---|
| `x` | Input tensor. |
| `dim` | Dimension along which to accumulate. |
| `output_dtype` | Optional output dtype. The operator default is FP32. |
| `r` | Optional output tensor with the same shape as `x`. |

**Returns**

- Cumulative-sum tensor with the same shape as `x`.

<h3 id="axdslsoftmax">ax.dsl.softmax</h3>

**签名**

```python
def softmax(
    x: Tensor,
    *,
    dim: Optional[int]=-1,
    output_dtype: typing_dtype=None,
    r: Optional[Tensor]=None,
) -> Tensor

```

**功能概述**

Computes softmax along a dimension.

For an input vector along the selected dimension:

$$\operatorname{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

| 参数 | 说明 |
|---|---|
| `x` | Input tensor. |
| `dim` | Softmax dimension. |
| `output_dtype` | Optional output dtype. |
| `r` | Optional output tensor. |

**Returns**

- Softmax result tensor.

<h3 id="axdsltopk">ax.dsl.topk</h3>

**签名**

```python
def topk(
    x: Tensor,
    *,
    k: int,
    dim: Optional[int]=-1,
    r_values: Optional[Tensor]=None,
    r_indices: Optional[Tensor]=None,
) -> Iterator[Tensor]

```

**功能概述**

Returns top-k values and indices along a dimension.

| 参数 | 说明 |
|---|---|
| `x` | Input tensor. |
| `k` | Number of elements to select. |
| `dim` | Selection dimension. |
| `r_values` | Optional values output tensor. |
| `r_indices` | Optional indices output tensor. |

**Returns**

- Top-k values tensor and top-k indices tensor.

<h3 id="axdslnonzero">ax.dsl.nonzero</h3>

**签名**

```python
def nonzero(
    x: Tensor,
    *,
    r_indices: Optional[Tensor]=None,
    r_num: Optional[Tensor]=None,
) -> Iterator[Tensor]

```

**功能概述**

Returns nonzero indices and the nonzero count.

If `x` has more than one dimension, the nonzero operation is applied to each row along the innermost dimension.

| 参数 | 说明 |
|---|---|
| `x` | Input tensor. |
| `r_indices` | Optional indices output tensor. |
| `r_num` | Optional count output tensor. |

**Returns**

- Nonzero indices tensor and nonzero count tensor.


## GEMM 与卷积
<h3 id="axdslmatmul">ax.dsl.matmul</h3>

**签名**

```python
def matmul(
    x: Tensor,
    y: Tensor,
    *,
    bias: Optional[np.ndarray]=None,
    w_scale: Union[float, Tensor]=1.0,
    w_scale2: Union[float, Tensor]=1.0,
    output_dtype: typing_dtype=None,
    transposed: Optional[bool]=False,
    output_add_assign: Optional[bool]=False,
    const_inputs: Optional[List[str]]=None,
    r: Optional[Tensor]=None,
) -> Tensor

```

**功能概述**

Matrix multiplication.

| 参数 | 说明 |
|---|---|
| `x` | Left input tensor. |
| `y` | Right input tensor. |
| `bias` | Optional constant bias. Deprecated. |
| `w_scale` | Weight scale. Shared scale for mx dtypes. |
| `w_scale2` | Secondary weight scale. Per tensor shared scale for nv dtypes. |
| `output_dtype` | Optional output dtype. |
| `transposed` | Whether `y` input is transposed. |
| `output_add_assign` | Whether to perform an in-place addition with the output tensor. |
| `const_inputs` | Specifies which input is constant. Deprecated. |
| `r` | Optional output tensor. |

**Returns**

- Result tensor.

<h3 id="axdslconv">ax.dsl.conv</h3>

**签名**

```python
def conv(
    x: Tensor,
    w: Union[np.ndarray, Tensor],
    b: Optional[np.ndarray]=None,
    *,
    stride: Optional[Sequence[int]]=None,
    padding: Optional[Sequence[int]]=None,
    dilation: Optional[Sequence[int]]=None,
    group: int=1,
    layout: str='NHWC',
    output_dtype: typing_dtype=None,
    w_scale: Union[float, np.ndarray]=1.0,
    w_scale2: Union[float, np.ndarray]=1.0,
    r: Optional[Tensor]=None,
) -> Tensor

```

**功能概述**

Record a convolution with constant or runtime-provided weights.

Equivalent to `torch.nn.functional.conv1d`, `conv2d`, or `conv3d`, selected by `layout` and the rank of `w`. `NLC` and `NCL` use `AxConvV2` when `w` is a runtime tensor; all other cases use `AxConv` with compile-time constants.

| 参数 | 说明 |
|---|---|
| `x` | Input tensor. |
| `w` | Convolution weight shaped `(OC, IC/group, *kernel_shape)`. NumPy weights are compile-time constants. A DSL tensor weight is supported for `NLC` and `NCL`. |
| `b` | Optional constant bias shaped `(OC,)`; defaults to zeros. |
| `stride` | Spatial stride. Defaults to ones. |
| `padding` | Spatial padding. Use `[left, right]` for 1D, four values for 2D, and six values for 3D. |
| `dilation` | Spatial dilation. Defaults to ones. |
| `group` | Convolution groups. |
| `layout` | One of `NLC`, `NCL`, `NHWC`, `NCHW`, `NDHWC`, or `NCDHW`. |
| `output_dtype` | Optional output dtype. Defaults to `x.dtype`. |
| `w_scale` | Optional constant-weight scale. |
| `w_scale2` | Optional secondary constant-weight scale. |
| `r` | Optional output tensor. |

**Returns**

- Convolution output tensor.


## Shape 与索引
<h3 id="axdslreshape">ax.dsl.reshape</h3>

**签名**

```python
def reshape(
    x: Tensor,
    *,
    shape: Tuple[int, ...],
    r: Optional[Tensor]=None,
) -> Tensor

```

**功能概述**

Reshapes a tensor to the target shape.

| 参数 | 说明 |
|---|---|
| `x` | Input tensor. |
| `shape` | Target shape. |
| `r` | Optional output tensor. |

**Returns**

- Reshaped tensor.

<h3 id="axdsltranspose">ax.dsl.transpose</h3>

**签名**

```python
def transpose(
    x: Tensor,
    *,
    perm: Tuple[int, ...],
    r: Optional[Tensor]=None,
) -> Tensor

```

**功能概述**

Permutes tensor dimensions.

Similar to `torch.permute(x, perm)`.

| 参数 | 说明 |
|---|---|
| `x` | Input tensor. |
| `perm` | Output dimension order. |
| `r` | Optional output tensor. |

**Returns**

- Transposed tensor.

<h3 id="axdslconcat">ax.dsl.concat</h3>

**签名**

```python
def concat(
    *xs: Tensor,
    dim: int,
    r: Optional[Tensor]=None,
) -> Tensor

```

**功能概述**

Concatenates tensors along a dimension.

| 参数 | 说明 |
|---|---|
| `*xs` | Input tensors to concatenate. |
| `dim` | Concatenation dimension. |
| `r` | Optional output tensor. |

**Returns**

- Concatenated tensor.

<h3 id="axdslexpand">ax.dsl.expand</h3>

**签名**

```python
def expand(
    x: Tensor,
    *,
    shape: Tuple[int, ...],
    r: Optional[Tensor]=None,
) -> Tensor

```

**功能概述**

Broadcasts a tensor to a target shape.

| 参数 | 说明 |
|---|---|
| `x` | Input tensor. |
| `shape` | Target broadcast shape. |
| `r` | Optional output tensor. |

**Returns**

- Broadcasted tensor with the target shape.

<h3 id="axdslpad">ax.dsl.pad</h3>

**签名**

```python
def pad(
    x: Tensor,
    *,
    padding: Tuple[int, ...],
    value: Optional[Union[int, float]]=0,
    r: Optional[Tensor]=None,
) -> Tensor

```

**功能概述**

Pads a tensor with the specified value.

Similar to torch.nn.functional.pad, but the dimensions specified by padding are ordered in reverse, from the outer dimension to the inner dimension.

| 参数 | 说明 |
|---|---|
| `x` | Input tensor. |
| `padding` | AxPad padding configuration. |
| `value` | Padding value. |
| `r` | Optional output tensor. |

**Returns**

- Padded tensor.

<h3 id="axdslgather">ax.dsl.gather</h3>

**签名**

```python
def gather(
    x: Tensor,
    indices: Tensor,
    *,
    dim: int,
    r: Optional[Tensor]=None,
) -> Tensor

```

**功能概述**

Gathers elements from a tensor along a dimension.

Similar to `torch.gather(x, dim, indices, out=r)`.

| 参数 | 说明 |
|---|---|
| `x` | Source tensor. |
| `indices` | Index tensor. |
| `dim` | Gather dimension. |
| `r` | Optional output tensor. |

**Returns**

- Tensor gathered from indexed positions.

<h3 id="axdslscatter">ax.dsl.scatter</h3>

**签名**

```python
def scatter(
    x: Tensor,
    indices: Tensor,
    *,
    dim: int,
    r: Tensor,
) -> Tensor

```

**功能概述**

Writes `x` into `r` at indexed positions.

Similar to `torch.scatter(r, dim, indices, x)`.

| 参数 | 说明 |
|---|---|
| `x` | Update tensor. |
| `indices` | Index tensor. |
| `dim` | Indexed dimension. |
| `r` | Target tensor to update. |

**Returns**

- Target tensor with indexed writes applied.

<h3 id="axdslindex_select">ax.dsl.index_select</h3>

**签名**

```python
def index_select(
    x: Tensor,
    indices: Tensor,
    *,
    dim: int,
    r: Optional[Tensor]=None,
) -> Tensor

```

**功能概述**

Selects slices from a tensor along a dimension.

Similar to `torch.index_select(x, dim, indices, out=r)`.

| 参数 | 说明 |
|---|---|
| `x` | Source tensor. |
| `indices` | Index tensor. |
| `dim` | Selection dimension. |
| `r` | Optional output tensor. |

**Returns**

- Tensor selected by indices along the dimension.

<h3 id="axdslindex_add">ax.dsl.index_add</h3>

**签名**

```python
def index_add(
    x: Tensor,
    indices: Tensor,
    *,
    dim: int,
    r: Tensor,
) -> Tensor

```

**功能概述**

Accumulates updates into a target tensor at indexed positions.

Similar to `torch.index_add(r, dim, indices, x, out=r)`. AXDSL uses r as both the target input and the output tensor.

| 参数 | 说明 |
|---|---|
| `x` | Update tensor to add. |
| `indices` | Index tensor. |
| `dim` | Indexed dimension. |
| `r` | Target tensor to be updated. |

**Returns**

- Target tensor with indexed additions applied.

<h3 id="axdslselect_scatter">ax.dsl.select_scatter</h3>

**签名**

```python
def select_scatter(
    x: Tensor,
    indices: Tensor,
    *,
    dim: int,
    r: Tensor,
) -> Tensor

```

**功能概述**

Embeds the values of the `x` into `r` at the given index.

Similar to `torch.select_scatter(r, x, dim, indices)`. AXDSL uses r as both the input and the output tensor.

| 参数 | 说明 |
|---|---|
| `x` | Update tensor. |
| `indices` | Index tensor. |
| `dim` | Indexed dimension. |
| `r` | Target tensor to update. |

**Returns**

- Target tensor with selected indexed writes applied.


## 结构化控制流
<h3 id="axdslif_op">ax.dsl.if_op</h3>

**签名**

```python
def if_op(
    cond: Tensor,
    r: typing.Optional[Tensor],
    true_func: callable,
    false_func: callable,
    *operands,
) -> typing.List[Tensor]

```

**功能概述**

Records conditional execution with true and false branch subgraphs.

| 参数 | 说明 |
|---|---|
| `cond` | Condition tensor. It must also be present in operands. |
| `r` | Optional output tensor. |
| `true_func` | Function traced for the true branch. |
| `false_func` | Function traced for the false branch. |
| `*operands` | Operands passed to both branch functions. |

**Returns**

- Conditional branch output tensors.

<h3 id="axdslwhile_loop">ax.dsl.while_loop</h3>

**签名**

```python
def while_loop(
    cond: Tensor,
    body_func: callable,
    *operands,
) -> typing.List[Tensor]

```

**功能概述**

Records a while-loop control-flow operation.

| 参数 | 说明 |
|---|---|
| `cond` | Loop-condition tensor, which must also be included in `operands`. The body function must update this tensor; the loop terminates when its value becomes zero. |
| `body_func` | Function to trace as the loop body. It receives the loop operands and returns their updated values in the same order. |
| `*operands` | Loop-carried tensors passed to the body and returned by the loop. |

**Returns**

- The updated loop-operand tensors in the same order as `operands`.

**Raises**

- AssertionError: If `cond` is not one of the loop operands.

<h3 id="axdslrange">ax.dsl.range</h3>

**签名**

```python
def range_loop(
    index: Tensor,
    start: Tensor,
    stop: Tensor,
    step: Tensor,
    body_func: callable,
    *operands,
    unroll_factor=1,
) -> typing.List[Tensor]

```

**功能概述**

Record one canonical counted loop.

The body is traced exactly once. `unroll_factor` is retained as a graph transformation request; lane cloning and remainder handling run only after the complete root graph has been recorded.

<h3 id="axdslswitch">ax.dsl.switch</h3>

**签名**

```python
def switch(
    selector: Tensor,
    branches: typing.Dict[int, typing.Callable],
    *operands,
) -> typing.List[Tensor]

```

**功能概述**

Record a switch with dense numeric cases.

This is an AXDSL-only control-flow API. Every branch is traced immediately into a child Graph, while runtime execution selects exactly one branch. Keys must be the dense range `0..N-1` because the backend uses `selector` as a branch-table index. A default branch is not supported.

| 参数 | 说明 |
|---|---|
| `selector` | Scalar tensor containing the runtime branch-table index. Its value must be in `0..len(branches)-1`. |
| `branches` | Mapping from dense integer case values to branch callables. Every callable receives `operands` and must return the same ordered tensor list. |
| `*operands` | Tensors passed to every branch callable while tracing. |

**Returns**

- The common ordered list of branch output tensors.

**Raises**

- AssertionError: If branch keys, callables, or outputs violate the switch contract.


## 调试
<h3 id="axdsldevice_print">ax.dsl.device_print</h3>

**签名**

```python
def device_print(
    prefix: str,
    t: Tensor,
    fn: Optional[Callable]=None,
) -> None

```

**功能概述**

Registers a device-side tensor print request.

The content of the tensor will be printed during simulation.

| 参数 | 说明 |
|---|---|
| `prefix` | Printed prefix. |
| `t` | Tensor to print. |
| `fn` | Optional Python callback. |

**Returns**

- None.


## 多核与多芯片并行
<h3 id="axparallelsync_all">ax.parallel.sync_all</h3>

**签名**

```python
def sync_all() -> None

```

**功能概述**

Record a compatibility synchronization marker.

AX8860 one-pass lowering does not turn this marker into a cross-core dependency. Use tensor-scoped `pub`/`sub` for dependencies between cores.

**Returns**

- None.

<h3 id="axparallelshmem_create_tensor">ax.parallel.shmem_create_tensor</h3>

**签名**

```python
def shmem_create_tensor(
    shape: typing.Tuple[int, ...],
    dtype: DslDtype,
    name: str=None,
) -> Tensor

```

**功能概述**

Creates a DDR tensor for current chip, based on openSHMEM communication model.

The signal should be created with `shmem_create_signal` api.

| 参数 | 说明 |
|---|---|
| `shape` | Shared tensor shape. |
| `dtype` | DSL dtype. |
| `name` | Optional tensor name. |

**Returns**

- Shared-memory tensor for the current chip.

<h3 id="axparallelshmem_create_signal">ax.parallel.shmem_create_signal</h3>

**签名**

```python
def shmem_create_signal(
    value: int=0,
) -> Tensor

```

**功能概述**

Creates a signal shared by multiple chips, based on openSHMEM communication model.

| 参数 | 说明 |
|---|---|
| `value` | initial signal value. |

**Returns**

- Signal tensor for the current chip.

<h3 id="axparallelshmem_put">ax.parallel.shmem_put</h3>

**签名**

```python
def shmem_put(
    dst: Tensor,
    src: Tensor,
    pe: int,
) -> None

```

**功能概述**

Copies a DDR tensor to a remote chip with a blocking put.

AXDSL-only communication API. Source and destination must be DDR tensors.

| 参数 | 说明 |
|---|---|
| `dst` | Remote DDR destination tensor. |
| `src` | Local DDR source tensor. |
| `pe` | Destination chip id. |

**Returns**

- None.

<h3 id="axparallelshmem_put_signal_nbi">ax.parallel.shmem_put_signal_nbi</h3>

**签名**

```python
def shmem_put_signal_nbi(
    dst: Tensor,
    src: Tensor,
    signal: Tensor,
    value: int,
    pe: int,
    sig_op: SigOp=SigOp.SET,
) -> None

```

**功能概述**

Issues a non-blocking put to a remote chip and updates a remote signal on completion, based on openSHMEM communication model.

| 参数 | 说明 |
|---|---|
| `dst` | Remote DDR destination tensor. |
| `src` | Local DDR source tensor. |
| `signal` | Signal updated on completion. |
| `value` | Signal update value. |
| `pe` | Destination chip id. |
| `sig_op` | Signal update operation. |

**Returns**

- None.

<h3 id="axparallelshmem_signal_update">ax.parallel.shmem_signal_update</h3>

**签名**

```python
def shmem_signal_update(
    signal: Tensor,
    value: int,
    pe: int,
    sig_op: SigOp,
) -> None

```

**功能概述**

Updates a signal on a remote chip, based on openSHMEM communication model.

| 参数 | 说明 |
|---|---|
| `signal` | Remote signal tensor to update. |
| `value` | Signal update value. |
| `pe` | Remote chip id. |
| `sig_op` | Signal update operation. |

**Returns**

- None.

<h3 id="axparallelsignal_update">ax.parallel.signal_update</h3>

**签名**

```python
def signal_update(
    signal: Tensor,
    value: int,
    sig_op: SigOp=SigOp.SET,
) -> None

```

**功能概述**

Updates a signal on the current chip.

| 参数 | 说明 |
|---|---|
| `signal` | Signal tensor to update. |
| `value` | Signal update value. |
| `sig_op` | Signal update operation. |

**Returns**

- None.

<h3 id="axparallelsignal_wait">ax.parallel.signal_wait</h3>

**签名**

```python
def signal_wait(
    signal: Tensor,
    value: int,
) -> None

```

**功能概述**

Waits until a local chip signal reaches a target value.

The barrier is set on current core (for computation kernel) or neulink (for communication kernel).

| 参数 | 说明 |
|---|---|
| `signal` | Signal tensor to wait on. |
| `value` | Target value. |

**Returns**

- None.

<h3 id="axparallelpub">ax.parallel.pub</h3>

**签名**

```python
def pub(
    tensor: Tensor,
) -> Tensor

```

**功能概述**

Publish the writes to `tensor` recorded before this point.

<h3 id="axparallelsub">ax.parallel.sub</h3>

**签名**

```python
def sub(
    tensor: Tensor,
) -> Tensor

```

**功能概述**

Subscribe subsequent readers to the latest visible publication.

<h3 id="axhostshmem_create_tensor">ax.host.shmem_create_tensor</h3>

**签名**

```python
def shmem_create_tensor(
    shape: typing.Tuple[int, ...],
    dtype: torch.dtype,
    name: str=None,
) -> Tensor

```

**功能概述**

Creates a cross-chip shared-memory tensor from host code.

AXDSL-only communication-buffer API. It registers shared DDR tensor metadata across chip contexts.

| 参数 | 说明 |
|---|---|
| `shape` | Shared tensor shape. |
| `dtype` | Torch dtype converted internally to a NumPy dtype. |
| `name` | Optional tensor name. |

**Returns**

- Shared tensor metadata.

<h3 id="axhostshmem_create_signal">ax.host.shmem_create_signal</h3>

**签名**

```python
def shmem_create_signal(
    value: int=0,
) -> Tensor

```

**功能概述**

Creates a cross-chip signal tensor from host code.

AXDSL-only synchronization API. The value argument is retained by the API but is not currently emitted as an initialization op.

| 参数 | 说明 |
|---|---|
| `value` | Requested initial signal value. |

**Returns**

- Signal tensor metadata.

