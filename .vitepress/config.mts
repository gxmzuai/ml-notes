import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "机器学习学习笔记",
  description: "机器学习学习笔记",
  head: [['link', { rel: 'icon', href: '/vitepress-logo-mini.png' }]],
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Notes', link: '/240308' }
    ],

    sidebar: [
      {
        text: 'Notes',
        items: [
          { text: '机器学习介绍', link: '/240308' },
          { text: '模型评估与选择', link: '/240315'}
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/gxmzuai/ml-notes' }
    ]
  }
})
