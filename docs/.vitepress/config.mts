import { defineConfig } from 'vitepress'

// Read the Docs serves each version under a path such as /en/latest/.
// Keep the root "/" for local preview and GitHub Pages deployment at the domain root.
const base = process.env.READTHEDOCS_CANONICAL_URL
  ? new URL(process.env.READTHEDOCS_CANONICAL_URL).pathname.replace(/\/$/, "")
  : "/"

export default defineConfig({
  base,
  lang: 'zh-CN',
  title: 'AXDSL 社区教程',
  description: '面向 AX8860 NPU 的 Python 自定义 Kernel DSL：从入门、内存模型、多核多芯片编程到性能调优。',
  head: [
    ['meta', { name: 'theme-color', content: '#0b66ff' }],
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/favicon.svg' }]
  ],
  themeConfig: {
    nav: [
      { text: '入门', link: '/guide/why-axdsl' },
      { text: '核心原理', link: '/guide/architecture' },
      { text: '进阶实战', link: '/advanced/operators' },
      { text: '教程', link: '/tutorials/vector-add' },
      { text: '参考', link: '/reference/operators' }
    ],
    sidebar: [
      {
        text: '开始',
        items: [
          { text: '首页与学习路径', link: '/' },
          { text: '为什么选择 AXDSL', link: '/guide/why-axdsl' },
          { text: '10 分钟快速上手', link: '/guide/quickstart' }
        ]
      },
      {
        text: '核心原理',
        items: [
          { text: '总体架构与工作原理', link: '/guide/architecture' },
          { text: '编程模型', link: '/guide/programming-model' },
          { text: '内存模型与数据搬运', link: '/guide/memory-model' },
          { text: '执行、验证与调试', link: '/guide/execution' }
        ]
      },
      {
        text: '进阶能力',
        items: [
          { text: '常用算子与 PyTorch 迁移', link: '/advanced/operators' },
          { text: '结构化控制流与动态执行', link: '/advanced/control-flow' },
          { text: '多核并行编程', link: '/advanced/multi-core' },
          { text: '多芯片与 Neulink 通信', link: '/advanced/multi-chip' },
          { text: '性能优化指南', link: '/advanced/performance' }
        ]
      },
      {
        text: '实战教程',
        items: [
          { text: 'Vector Add：第一个 Kernel', link: '/tutorials/vector-add' },
          { text: 'Tiled MatMul：二维切分', link: '/tutorials/tiled-matmul' },
          { text: 'Fused SiLU × Mul：算子融合', link: '/tutorials/fused-silu-mul' },
          { text: 'Dynamic Loop：动态循环', link: '/tutorials/dynamic-loop' },
          { text: 'Ring All-Gather：多芯片通信', link: '/tutorials/ring-all-gather' }
        ]
      },
      {
        text: '参考手册',
        items: [
          { text: '完整算子列表', link: '/reference/operators' },
          { text: 'API Cookbook', link: '/reference/cookbook' },
          { text: 'FAQ 与故障排查', link: '/reference/faq' },
          { text: '术语表', link: '/reference/glossary' },
          { text: '发布与维护', link: '/reference/release-and-maintenance' }
        ]
      }
    ],
    search: {
      provider: 'local',
      options: {
        translations: {
          button: { buttonText: '搜索文档', buttonAriaLabel: '搜索文档' },
          modal: {
            noResultsText: '没有找到结果',
            resetButtonTitle: '清除查询',
            displayDetails: '显示详情',
            footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' }
          }
        }
      }
    },
    outline: { level: [2, 3], label: '本页目录' },
    docFooter: { prev: '上一篇', next: '下一篇' },
    returnToTopLabel: '回到顶部',
    externalLinkIcon: true,
    socialLinks: []
  },
  markdown: { lineNumbers: true, math: true },
  lastUpdated: true,
  vite: { logLevel: 'info' }
})
