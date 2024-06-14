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
          { text: '模型评估与选择', link: '/240315' },
          { text: '模型训练', link: '/240322' },
          { text: 'CNN && RNN', link: '/240329' },
          { text: 'RNN、LSTM、GRU1', link: '/240407' },
          { text: 'RNN、LSTM、GRU2', link: '/240419' },
          { text: '词向量、seq2seq', link: '/240426' },
          { text: 'transformer1', link: '/240510' },
          { text: 'transformer2', link: '/240517' },
          { text: '上机课1', link: '/240524' },
          { text: '上机课2', link: '/240531' },
          { text: '上机课3', link: '/240607' },
          { text: '上机课4', link: '/240614' }
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/gxmzuai/ml-notes' }
    ]
  }
})
