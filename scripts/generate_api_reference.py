"""Generate the AXDSL operator/API reference page from source docstrings."""

import ast
import os
import re
from pathlib import Path

REPO = Path(os.environ.get("AXDSL_API_ROOT", Path.home() / "npu-codebase" / "axnn" / "axnn" / "dsl" / "api")).expanduser()
OUT = Path(__file__).parents[1] / "docs" / "reference" / "operators.md"

GROUPS = [
    ("Kernel 与上下文", [
        ("ax.builder", "builder.py", "builder"),
        ("ax.init", "init.py", "init"),
        ("ax.program_id", "program.py", "program_id"),
        ("ax.num_programs", "program.py", "num_programs"),
        ("ax.init_session", "../device_npu/session.py", "init_session"),
        ("ax.synchronize_stream", "../device_npu/session.py", "synchronize_stream"),
        ("ax.finalize", "../device_npu/session.py", "finalize"),
    ]),
    ("内存与数据搬运", [
        ("ax.dsl.load", "lang/load.py", "load"),
        ("ax.dsl.store", "lang/store.py", "store"),
        ("ax.dsl.move", "lang/move.py", "move"),
        ("ax.dsl.dynamic_ld", "lang/dynamic_ld.py", "dynamic_ld"),
        ("ax.dsl.dynamic_st", "lang/dynamic_st.py", "dynamic_st"),
    ]),
    ("Tensor 创建与常量", [
        ("ax.dsl.create_tensor", "lang/create_tensor.py", "create_tensor"),
        ("ax.dsl.zeros", "lang/zeros.py", "zeros"),
        ("ax.dsl.full", "lang/full.py", "full"),
        ("ax.dsl.from_numpy", "lang/from_numpy.py", "from_numpy"),
        ("ax.dsl.cast", "lang/cast.py", "cast"),
    ]),
    ("Elementwise 与数学", [
        ("ax.dsl.add", "lang/add.py", "add"),
        ("ax.dsl.sub", "lang/sub.py", "sub"),
        ("ax.dsl.mul", "lang/mul.py", "mul"),
        ("ax.dsl.div", "lang/div.py", "div"),
        ("ax.dsl.maximum", "lang/maximum.py", "maximum"),
        ("ax.dsl.minimum", "lang/minimum.py", "minimum"),
        ("ax.dsl.clip", "lang/clip.py", "clip"),
        ("ax.dsl.where", "lang/where.py", "where"),
        ("ax.dsl.exp", "lang/exp.py", "exp"),
        ("ax.dsl.log2", "lang/log2.py", "log2"),
        ("ax.dsl.sqrt", "lang/sqrt.py", "sqrt"),
        ("ax.dsl.reciprocal", "lang/reciprocal.py", "reciprocal"),
        ("ax.dsl.floor", "lang/floor.py", "floor"),
        ("ax.dsl.ceil", "lang/ceil.py", "ceil"),
    ]),
    ("激活函数", [
        ("ax.dsl.sigmoid", "lang/sigmoid.py", "sigmoid"),
        ("ax.dsl.tanh", "lang/tanh.py", "tanh"),
        ("ax.dsl.gelu", "lang/gelu.py", "gelu"),
        ("ax.dsl.quick_gelu", "lang/quick_gelu.py", "quick_gelu"),
        ("ax.dsl.silu", "lang/silu.py", "silu"),
        ("ax.dsl.softplus", "lang/softplus.py", "softplus"),
    ]),
    ("Reduction 与搜索", [
        ("ax.dsl.sum", "lang/sum.py", "sum"),
        ("ax.dsl.max", "lang/max.py", "max"),
        ("ax.dsl.min", "lang/min.py", "min"),
        ("ax.dsl.argmax", "lang/argmax.py", "argmax"),
        ("ax.dsl.argmin", "lang/argmin.py", "argmin"),
        ("ax.dsl.cumsum", "lang/cumsum.py", "cumsum"),
        ("ax.dsl.softmax", "lang/softmax.py", "softmax"),
        ("ax.dsl.topk", "lang/topk.py", "topk"),
        ("ax.dsl.nonzero", "lang/nonzero.py", "nonzero"),
    ]),
    ("GEMM 与卷积", [
        ("ax.dsl.matmul", "lang/matmul.py", "matmul"),
        ("ax.dsl.conv", "lang/conv.py", "conv"),
    ]),
    ("Shape 与索引", [
        ("ax.dsl.reshape", "lang/reshape.py", "reshape"),
        ("ax.dsl.transpose", "lang/transpose.py", "transpose"),
        ("ax.dsl.concat", "lang/concat.py", "concat"),
        ("ax.dsl.expand", "lang/expand.py", "expand"),
        ("ax.dsl.pad", "lang/pad.py", "pad"),
        ("ax.dsl.gather", "lang/gather.py", "gather"),
        ("ax.dsl.scatter", "lang/scatter.py", "scatter"),
        ("ax.dsl.index_select", "lang/index_select.py", "index_select"),
        ("ax.dsl.index_add", "lang/index_add.py", "index_add"),
        ("ax.dsl.select_scatter", "lang/select_scatter.py", "select_scatter"),
    ]),
    ("结构化控制流", [
        ("ax.dsl.if_op", "control_flow/if_op.py", "if_op"),
        ("ax.dsl.while_loop", "control_flow/while_loop.py", "while_loop"),
        ("ax.dsl.range", "control_flow/range.py", "range_loop"),
        ("ax.dsl.switch", "control_flow/switch.py", "switch"),
    ]),
    ("调试", [
        ("ax.dsl.device_print", "lang/device_print.py", "device_print"),
    ]),
    ("多核与多芯片并行", [
        ("ax.parallel.sync_all", "parallel/sync_all.py", "sync_all"),
        ("ax.parallel.shmem_create_tensor", "parallel/shmem_create_tensor.py", "shmem_create_tensor"),
        ("ax.parallel.shmem_create_signal", "parallel/shmem_create_signal.py", "shmem_create_signal"),
        ("ax.parallel.shmem_put", "parallel/shmem_put.py", "shmem_put"),
        ("ax.parallel.shmem_put_signal_nbi", "parallel/shmem_put_signal_nbi.py", "shmem_put_signal_nbi"),
        ("ax.parallel.shmem_signal_update", "parallel/shmem_signal_update.py", "shmem_signal_update"),
        ("ax.parallel.signal_update", "parallel/signal_update.py", "signal_update"),
        ("ax.parallel.signal_wait", "parallel/signal_wait.py", "signal_wait"),
        ("ax.parallel.pub", "parallel/pub_sub.py", "pub"),
        ("ax.parallel.sub", "parallel/pub_sub.py", "sub"),
        ("ax.host.shmem_create_tensor", "host/shmem_create_tensor.py", "shmem_create_tensor"),
        ("ax.host.shmem_create_signal", "host/shmem_create_signal.py", "shmem_create_signal"),
    ]),
]


def get_func(rel: str, name: str):
    path = REPO / rel
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise KeyError(f"{name} not found in {path}")


def split_top_level(text: str) -> list[str]:
    """Split an AST-unpacked argument list on top-level commas."""
    parts: list[str] = []
    start_i = 0
    depth = 0
    quote: str | None = None
    escaped = False
    for i, ch in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start_i:i].strip())
            start_i = i + 1
    tail = text[start_i:].strip()
    if tail:
        parts.append(tail)
    return parts


def signature(node) -> str:
    """Return a readable, wrapped Python signature."""
    try:
        args_text = ast.unparse(node.args)
        returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    except Exception:
        return f"def {node.name}(...)"

    args = split_top_level(args_text)
    if not args:
        return f"def {node.name}(){returns}"
    wrapped = [f"def {node.name}("]
    for i in range(0, len(args), 4):
        wrapped.extend(f"    {part}," for part in args[i : i + 4])
    wrapped.append(f"){returns}")
    return "\n".join(wrapped)

def clean_doc(node) -> str:
    return (ast.get_docstring(node) or "暂无源码 docstring。").strip()


def inline_rst(text: str) -> str:
    """Render common inline RST markup used in AXDSL source docstrings."""
    text = text.strip()
    text = text.replace("``", "`")
    text = re.sub(r":math:`([^`]+)`", r"$\1$", text)
    return text


def normalize_description(lines: list[str]) -> list[str]:
    """Normalize paragraphs, display math, and literal blocks in descriptions."""
    result: list[str] = []
    paragraph: list[str] = []
    math_lines: list[str] = []

    def flush_paragraph():
        if paragraph:
            result.append(inline_rst(" ".join(paragraph)))
            paragraph.clear()

    for raw in lines:
        line = raw.rstrip()
        if line.startswith(".. math::"):
            flush_paragraph()
            math_lines = [line[len(".. math::"):].strip()]
            continue
        if math_lines:
            if not line.strip():
                math_lines.append("")
            elif line.startswith(" "):
                math_lines.append(line.strip())
            else:
                formula = " ".join(x for x in math_lines if x)
                if formula:
                    result.append(f"$${formula}$$")
                math_lines = []
                paragraph.append(line.strip())
            continue
        if not line.strip():
            flush_paragraph()
        else:
            paragraph.append(line.strip())

    if math_lines:
        formula = " ".join(x for x in math_lines if x)
        if formula:
            result.append(f"$${formula}$$")
    flush_paragraph()
    return result


def split_doc_sections(doc: str):
    """Split a Google-style docstring into ordered sections."""
    markers = {
        "Args:": "Args",
        "Returns:": "Returns",
        "Raises:": "Raises",
        "Yields:": "Yields",
        "Note:": "Note",
        "Notes:": "Notes",
        "Example:": "Example",
        "Examples:": "Examples",
    }
    sections: dict[str, list[str]] = {}
    order: list[tuple[str, list[str]]] = []
    current_name = "_description"
    current: list[str] = []

    for line in doc.splitlines():
        stripped = line.strip()
        if stripped in markers:
            if current or current_name != "_description":
                sections[current_name] = current
                order.append((current_name, current))
            current_name = markers[stripped]
            current = []
        else:
            current.append(line)

    sections[current_name] = current
    order.append((current_name, current))
    return sections, order


def parse_args(lines: list[str]) -> list[tuple[str, str]]:
    args: list[tuple[str, str]] = []
    current_name = ""
    current_body: list[str] = []

    def flush():
        if current_name:
            args.append((current_name, " ".join(x.strip() for x in current_body if x.strip())))
            current_body.clear()

    for line in lines:
        if not line.strip():
            continue
        # Google docstrings in this source use either 4-space or 8-space indent.
        if line.startswith("        ") and current_name and (":" in line):
            # Continuation that itself contains a colon is still description unless
            # the prefix is a known argument-like identifier followed by colon.
            candidate = line.strip().split(":", 1)[0]
            if candidate and " " not in candidate and not candidate.startswith(("e.g", "i.e")):
                flush()
                current_name = candidate
                current_body = [line.strip().split(":", 1)[1].strip()]
                continue
        if line.startswith("    ") and ":" in line:
            flush()
            name, body = line.strip().split(":", 1)
            current_name = name.strip()
            current_body = [body.strip()]
        elif current_name:
            current_body.append(line.strip())
    flush()
    return args


def parse_items(lines: list[str]) -> list[str]:
    items: list[str] = []
    current: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        stripped = line.strip()
        starts_item = bool(re.match(r"^[A-Za-z_][A-Za-z0-9_.]*:", stripped))
        if current and not starts_item and line.startswith(("    ", "        ")):
            current.append(stripped)
        else:
            if current:
                items.append(" ".join(current))
                current = []
            current = [stripped]
    if current:
        items.append(" ".join(current))
    return [inline_rst(x) for x in items]


def markdown_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render_doc(node) -> str:
    doc = clean_doc(node)
    sections, order = split_doc_sections(doc)
    out: list[str] = []

    for name, lines in order:
        if not any(x.strip() for x in lines):
            continue
        if name == "_description":
            out.extend(normalize_description(lines))
        elif name == "Args":
            args = parse_args(lines)
            if args:
                rows = [
                    f"| `{arg_name}` | {markdown_escape(inline_rst(desc))} |"
                    for arg_name, desc in args
                ]
                out.append("| 参数 | 说明 |\n|---|---|\n" + "\n".join(rows))
        elif name in ("Returns", "Yields"):
            values = parse_items(lines)
            out.append(f"**{name}**\n\n" + "\n".join(f"- {x}" for x in values))
        elif name == "Raises":
            values = parse_items(lines)
            out.append("**Raises**\n\n" + "\n".join(f"- {x}" for x in values))
        elif name in ("Note", "Notes"):
            values = parse_items(lines)
            out.append("**Note**\n\n" + "\n".join(f"- {x}" for x in values))
        elif name in ("Example", "Examples"):
            out.append(f"**{name}**\n\n```python\n" + "\n".join(x.strip() for x in lines if x.strip()) + "\n```")
        else:
            out.extend(normalize_description(lines))

    return "\n\n".join(x for x in out if x.strip())


heading = """# 完整算子与 API 列表

本页按能力列出当前源码公开导出的 AXDSL API。签名与说明直接来自 `axnn/axnn/dsl` 源码 docstring，便于与工具链版本对齐。教程中常用的语义解释见 [常用算子与 PyTorch 迁移](/advanced/operators)。

## 快速能力索引

| 类别 | API |
|---|---|
"""
for title, items in GROUPS:
    names = "<br>".join(f"[`{name}`](#{name.replace('.', '').lower()})" for name, _, _ in items)
    heading += f"| {title} | {names} |\n"

heading += """
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
"""

lines = [heading]
for group_title, items in GROUPS:
    lines.append(f"\n## {group_title}\n")
    for display_name, rel, func_name in items:
        node = get_func(rel, func_name)
        anchor = display_name.replace(".", "").lower()
        lines.append(f'<h3 id="{anchor}">{display_name}</h3>\n\n')
        lines.append("**签名**\n\n")
        lines.append("```python\n" + signature(node) + "\n\n```\n\n")
        lines.append("**功能概述**\n\n")
        lines.append(render_doc(node) + "\n\n")

OUT.write_text("".join(lines), encoding="utf-8")
print(f"generated {OUT} with {sum(len(x[1]) for x in GROUPS)} API entries")
