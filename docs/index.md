---
layout: home

hero:
  name: AXDSL
  text: AX8860 NPU 上的 Python Kernel DSL
  tagline: 用熟悉的 Python 表达自定义算子，显式掌控数据搬运、多核协作、多芯片通信与性能调优。
  actions:
    - theme: brand
      text: 10 分钟快速上手
      link: /guide/quickstart
    - theme: alt
      text: 理解工作原理
      link: /guide/architecture
    - theme: alt
      text: 查看算子能力
      link: /reference/operators

features:
  - icon: 🧠
    title: Python Kernel 心智模型
    details: 用 @ax.builder 将 Python 函数记录为 AXDSL Graph IR，保留 Kernel 开发直觉，同时面向 NPU 编译执行。
  - icon: 🚚
    title: 显式内存与数据搬运
    details: load / store / move 清楚表达 DDR、OCM、Shared OCM 之间的数据流，避免黑盒式性能不可控。
  - icon: 🧩
    title: 结构化控制流
    details: if、while、range、switch 与 dynamic_ld/st 让动态 shape 和数据相关执行不再是静态图的表达瓶颈。
  - icon: 🧵
    title: 多核与多芯片原生支持
    details: Program Grid 映射并行任务，SHARED_OCM、signal 和 Neulink API 支持 4 core 与多 chip 协同。
  - icon: 🧪
    title: PYRUN / CRUN / BRUN 工作流
    details: 先验证语义，再编译仿真，最后设备运行；支持输出与逐层结果检查。
  - icon: 📈
    title: 可观测的性能分析
    details: 导出 graph、job metadata、NPU trace、cycle trace 与 HTML 报告，用数据定位瓶颈。
---

## 推荐学习路径

| 目标 | 推荐路径 | 时间 |
|---|---|---|
| 跑通第一个 Kernel | [快速上手](/guide/quickstart) → [Vector Add](/tutorials/vector-add) | 10–30 分钟 |
| 理解核心模型 | [架构](/guide/architecture) → [编程模型](/guide/programming-model) → [内存模型](/guide/memory-model) | 1–2 小时 |
| 编写真实算子 | [常用算子](/advanced/operators) → [Tiled MatMul](/tutorials/tiled-matmul) → [性能优化](/advanced/performance) | 半天 |
| 掌握动态执行 | [控制流](/advanced/control-flow) → [Dynamic Loop](/tutorials/dynamic-loop) | 半天 |
| 进入多芯片 | [多核](/advanced/multi-core) → [多芯片](/advanced/multi-chip) → [Ring All-Gather](/tutorials/ring-all-gather) | 1 天 |

## 这套文档适合谁

- **模型 / 推理工程师**：希望把 PyTorch reference 中被证明有效的 fusion 或 shape 专用优化落到 NPU。
- **Kernel 工程师**：希望用更高层表达方式控制 tile、内存层级、任务划分和通信。
- **框架接入工程师**：需要理解 AXDSL 的输入输出、编译产物、缓存与运行模式。
- **社区开发者**：希望系统了解 AXDSL 能力边界，并参与示例、算子和工具链贡献。

## 文档状态

本站是面向社区推广的教程初稿，基于当前 AXDSL 中文文档、`axnn/axnn/dsl` 源码和内置示例整理。硬件能力、后端版本和 API 支持可能随工具链变化，正式发布前请以目标工具链的实际 API 签名为准。
