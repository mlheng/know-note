# 📚 KnowNote - 智能笔记助手

将零散笔记转化为**交互式知识图谱**，让学习更高效。

## ✨ 功能

- **文本笔记 → 知识图谱**：输入文字，自动提取关键词、构建关系图谱
- **Markdown 编辑**：支持 Markdown 格式的结构化笔记
- **图片 OCR 识别**：上传图片自动识别文字并生成图谱
- **批量导入**：一次导入多个 .txt/.md 文件，合并分析
- **AI 摘要**：可选接入 LLM（OpenAI / DeepSeek / 百度文心），自动生成笔记摘要
- **图谱节点点击查询**：🆕 点击图谱中的任意节点，AI 自动生成该关键词的详细解释
- **图谱导出**：将知识图谱导出为独立 HTML 文件

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动应用

```bash
streamlit run app.py
```

浏览器自动打开 `http://localhost:8501` 即可使用。

### 4. （推荐）部署到 Streamlit Cloud

让别人通过网址直接访问：

1. 将项目推送到 **GitHub 公开仓库**
2. 打开 [share.streamlit.io](https://share.streamlit.io)
3. 点击「New app」→ 选择仓库 → 填写：
   - **Main file path**: `app.py`
   - **Python version**: 3.10
4. 点击 Deploy，等待 2-3 分钟即可获得公网链接 🎉

> ⚠️ **注意**：Streamlit Cloud 免费版只有 1GB 内存，OCR 功能（PaddleOCR）可能会因内存不足而失败，其他功能正常。

### 5. （可选）启用 AI 摘要

在侧边栏勾选「启用 AI 摘要」并输入 API Key：

| 平台 | 获取 Key | 费用 |
|------|----------|------|
| DeepSeek | [platform.deepseek.com](https://platform.deepseek.com) | 极低 |
| OpenAI | [platform.openai.com](https://platform.openai.com) | 按量付费 |
| 百度文心 | [console.bce.baidu.com](https://console.bce.baidu.com) | 有免费额度 |

## 📁 项目结构

```
PythonProject2/
├── app.py                      # 主程序入口 (Streamlit UI)
├── requirements.txt            # Python 依赖清单
├── packages.txt                # Streamlit Cloud 系统依赖
├── README.md                   # 本文件
├── .gitignore                  # Git 忽略规则
└── modules/
    ├── __init__.py             # 模块导出
    ├── nlp_processor.py        # NLP 处理（分词、关键词提取、共现矩阵）
    ├── graph_builder.py        # 知识图谱构建（NetworkX → ECharts）
    ├── ocr_processor.py        # OCR 图片文字识别（PaddleOCR）
    └── llm_processor.py        # LLM 调用（OpenAI / DeepSeek / 百度）
```

## 🛠️ 技术栈

| 组件 | 技术 |
|------|------|
| Web 框架 | Streamlit |
| 中文分词 | jieba (TextRank / TF-IDF) |
| 图计算 | NetworkX |
| 可视化 | ECharts（力导向布局） |
| OCR | PaddleOCR |
| AI 摘要 | OpenAI / DeepSeek / 百度文心 API |
