# AXDSL 社区教程

[![Documentation Status](https://readthedocs.org/projects/axdsl-community-tutorial/badge/?version=latest)](https://axdsl-community-tutorial.readthedocs.io/zh-cn/latest/)

**在线阅读：** <https://axdsl-community-tutorial.readthedocs.io/zh-cn/latest/>

面向 AX8860 NPU 的中文 AXDSL 教程，基于 VitePress 构建。

## 本地开发

```bash
npm ci
npm run dev
```

访问：

```text
http://localhost:5177
```

## 构建与预览

```bash
npm run build
npm run preview
```

访问：

```text
http://localhost:4177
```

静态产物：

```text
docs/.vitepress/dist
```

## 内容结构

```text
docs/
├── index.md                     首页与学习路径
├── guide/                       入门与核心原理
├── advanced/                    控制流、多核、多芯片、性能
├── tutorials/                   完整实战教程
├── reference/                   API、Cookbook、FAQ、术语表、发布流程
├── examples/                    可运行 PYRUN 示例
└── public/                      favicon 等静态资源
```

## 示例运行

示例假设可以 import 当前 AXNN 工具链。可在 AXNN 仓库根目录执行官方示例，或按需设置 `PYTHONPATH`：

```bash
export PYTHONPATH="$HOME/npu-codebase/axnn"
python3 docs/examples/vector_add.py
python3 docs/examples/tiled_matmul.py
python3 docs/examples/fused_silu_mul.py
python3 docs/examples/dynamic_range.py
```

`docs/examples/ring_all_gather.py` 需要 CRUN 与多芯片仿真工具链支持。

## 更新 API Reference

API Reference 由源码 docstring 生成：

```bash
export AXDSL_API_ROOT=/path/to/npu-codebase/axnn/axnn/dsl/api
python3 scripts/generate_api_reference.py
```

提交生成后的 `docs/reference/operators.md`。Read the Docs 不依赖内部源码仓库，只构建仓库中已提交的 Markdown。

## GitHub 与 Read the Docs 发布

仓库根目录包含：

```text
.readthedocs.yaml
.github/workflows/docs.yml
```

推送到 GitHub 后：

1. GitHub Actions 自动执行 `npm ci && npm run build`
2. 在 Read the Docs 导入该仓库
3. Read the Docs 自动识别 `.readthedocs.yaml`
4. 默认公网地址类似：

```text
https://<project-slug>.readthedocs.io/en/latest/
```

详细步骤见 [发布与维护流程](docs/reference/release-and-maintenance.md)。

## 当前注意事项

- 当前源码公开 `ax.builder` 签名未包含旧文档提到的 `auto_tune_cases`，教程不将其作为可用能力承诺。
- 当前 CRUN 多芯片路径要求每 chip 1 core，文档按当前源码标注。
- dtype / shape 的算子级支持矩阵仍需目标工具链补充。
- 本项目采用 [BSD-3-Clause](LICENSE) 开源协议。

## License

This project is licensed under the [BSD 3-Clause License](LICENSE).
