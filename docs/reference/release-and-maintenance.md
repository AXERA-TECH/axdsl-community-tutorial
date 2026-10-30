# 文档发布与维护流程

本文说明如何把 AXDSL 教程发布到 GitHub，并通过 Read the Docs 自动构建和公开访问。

## 1. 仓库结构

```text
.
├── .github/workflows/docs.yml   # GitHub Actions 构建检查
├── .readthedocs.yaml             # Read the Docs 构建配置
├── docs
│   ├── .vitepress/config.mts     # VitePress 配置
│   ├── guide                     # 入门与核心原理
│   ├── advanced                  # 进阶能力
│   ├── tutorials                 # 实战教程
│   ├── reference                 # API、Cookbook、FAQ、术语表
│   ├── examples                  # 教程示例
│   └── public                    # 静态资源
├── scripts
│   └── generate_api_reference.py # 从 AXDSL 源码生成 API Reference
├── LICENSE                      # BSD-3-Clause
├── package.json
└── package-lock.json
```

## 2. 本地修改与验证

```bash
npm ci
npm run build
npm run preview
```

本地预览地址：

```text
http://localhost:4177
```

## 3. 更新 API Reference

`docs/reference/operators.md` 是生成文件，建议不要手工修改。更新方式：

```bash
export AXDSL_API_ROOT=/path/to/npu-codebase/axnn/axnn/dsl/api
python3 scripts/generate_api_reference.py
```

生成后检查 diff，并提交：

```bash
git add docs/reference/operators.md
git commit -m "docs: refresh AXDSL API reference"
```

::: note
当前生成脚本需要能访问 AXDSL 源码，因此 API Reference 在维护者本地或内部 CI 中生成。Read the Docs 只负责构建已提交的 Markdown，不依赖内部源码仓库。
:::

## 4. 推送到 GitHub

示例：

```bash
git init -b main
git add .
git commit -m "docs: initialize AXDSL community tutorial"
git remote add origin git@github.com:AXERA-TECH/axdsl-docs.git
git push -u origin main
```

建议工作流：

```text
main        最新稳定文档
feature/*   新章节或大改
fix/*       文档修复
```

大改动使用 Pull Request，并等待 GitHub Actions 构建通过。

## 5. 接入 Read the Docs

1. 打开 <https://readthedocs.org>。
2. 使用 GitHub 账号登录。
3. 进入 **My Projects → Import Project**。
4. 选择 GitHub 上的 `axdsl-docs` 仓库。
5. Read the Docs 会自动识别根目录的 `.readthedocs.yaml`。
6. 默认发布地址类似：

```text
https://<project-slug>.readthedocs.io/en/latest/
```

## 6. Read the Docs 构建逻辑

`.readthedocs.yaml` 执行：

```bash
npm ci
npm run build
cp -R docs/.vitepress/dist/. "$READTHEDOCS_OUTPUT/html/"
```

由于 Read the Docs 会把文档挂在 `/en/latest/`、`/en/stable/` 等子路径下，VitePress 配置中会读取：

```text
READTHEDOCS_CANONICAL_URL
```

并自动设置 `base`，避免 CSS、JS 和站内链接指向错误的根路径。

## 7. 版本策略

推荐：

| 版本 | 来源 | 用途 |
|---|---|---|
| `latest` | `main` | 持续更新 |
| `stable` | 最新 Git tag | 对外稳定入口 |
| `vX.Y` | 历史 tag | 归档查询 |

发布 stable：

```bash
git tag v1.0
git push origin v1.0
```

然后在 Read the Docs 项目后台的 **Versions** 中激活该 tag，并可将其设为 stable。

## 8. 常见问题

### 为什么不用 Sphinx？

参考项目 `pulsar2-docs` 使用 Sphinx，而本项目已经采用 VitePress。Read the Docs 支持任意生成 HTML 的文档工具，因此不需要迁移技术栈。

### 会自动更新吗？

会。推送到 GitHub 后，Read the Docs 会按项目配置自动触发构建。Pull Request 也可以启用预览构建。

### 可以绑定自定义域名吗？

可以。在 Read the Docs 项目后台的 **Domains** 中配置自定义域名，并按提示添加 DNS CNAME 记录。

### API 生成脚本在 Read the Docs 上会运行吗？

不会。`.readthedocs.yaml` 只执行 `npm ci` 和 `npm run build`。API Markdown 已提交到仓库，避免 Read the Docs 依赖内部 AXDSL 源码。

## 9. License

本项目采用 BSD-3-Clause 开源协议，见根目录 [LICENSE](https://github.com/AXERA-TECH/axdsl-docs/blob/main/LICENSE)。
