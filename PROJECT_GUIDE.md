# 📚 KnowNote 项目完全解读

> 读完本文，你就能理解这个项目的每一行代码在做什么，以及它们是如何协作的。

---

## 目录

1. [这个项目是干什么的](#1-这个项目是干什么的)
2. [项目文件结构](#2-项目文件结构)
3. [数据是如何流转的（核心流程）](#3-数据是如何流转的核心流程)
4. [逐文件深度解析](#4-逐文件深度解析)
   - [4.1 `app.py` — 用户界面](#41-apppy--用户界面)
   - [4.2 `modules/nlp_processor.py` — NLP 处理器](#42-modulesnlp_processorpy--nlp-处理器)
   - [4.3 `modules/graph_builder.py` — 图谱构建器](#43-modulesgraph_builderpy--图谱构建器)
   - [4.4 `modules/ocr_processor.py` — OCR 处理器](#44-modulesocr_processorpy--ocr-处理器)
   - [4.5 `modules/llm_processor.py` — 大模型处理器](#45-modulesllm_processorpy--大模型处理器)
   - [4.6 `modules/__init__.py` — 模块入口](#46-modules__init__py--模块入口)
   - [4.7 `requirements.txt` — 依赖清单](#47-requirementstxt--依赖清单)
5. [如何运行](#5-如何运行)
6. [如何扩展](#6-如何扩展)

---

## 1. 这个项目是干什么的

**KnowNote（智能笔记助手）** 是一个 Web 应用，它做的事情很简单：

```
你输入一段文字  →  自动提取关键词  →  生成一个可拖拽的知识图谱
```

比如你输入一段关于"人工智能"的笔记：

```
人工智能是计算机科学的一个分支，机器学习是人工智能的核心领域，
深度学习是机器学习的一个重要方法。
```

它会自动：
- 提取出关键词：**人工智能**、**机器学习**、**深度学习**、**计算机科学**
- 分析关键词之间的关系（哪些词经常一起出现）
- 生成一个**力导向图**（节点=关键词，连线=关联关系），你可以在图上拖拽、缩放

此外还支持：
- 📄 Markdown 格式笔记
- 🖼️ 上传图片自动 OCR 识别文字
- 📁 批量导入多个文件
- 🤖 接入 AI（OpenAI/DeepSeek/百度）自动生成摘要

---

## 2. 项目文件结构

```
PythonProject2/
│
├── app.py                          # 🚪 主入口：用户界面（Streamlit）
├── requirements.txt                # 📦 Python 依赖包清单
├── README.md                       # 📖 项目简介
│
└── modules/                        # 🧩 业务逻辑模块（核心引擎）
    ├── __init__.py                 #   包的入口，声明对外暴露哪些类
    ├── nlp_processor.py            #   🔤 中文分词 + 关键词提取
    ├── graph_builder.py            #   🕸️ 知识图谱的构建与渲染
    ├── ocr_processor.py            #   👁️ 图片文字识别（OCR）
    └── llm_processor.py            #   🤖 大模型 API 调用
```

**设计原则**：`app.py` 只负责"画界面"，`modules/` 只负责"干实事"。界面层和逻辑层完全分离，模块不依赖 Streamlit，可以单独拿出来用在其他地方。

---

## 3. 数据是如何流转的（核心流程）

当你点击「生成知识图谱」按钮时，数据经历了以下 5 个步骤：

```
┌──────────┐     ┌──────────────┐     ┌───────────────┐     ┌──────────────┐     ┌─────────────┐
│ 用户输入  │ ──→ │ NLPProcessor │ ──→ │ GraphBuilder  │ ──→ │  ECharts     │ ──→ │  浏览器展示  │
│ 一段文字  │     │ 分词+关键词   │     │ 构建图结构     │     │ HTML生成     │     │  交互式图谱  │
└──────────┘     └──────────────┘     └───────────────┘     └──────────────┘     └─────────────┘
                         │                                            │
                         ▼                                            ▼
                  ┌──────────────┐                          ┌────────────────┐
                  │ LLMProcessor │                          │ st.session_state│
                  │ AI 摘要(可选) │                          │ 缓存结果        │
                  └──────────────┘                          └────────────────┘
```

具体每一步发生了什么：

| 步骤 | 谁做的 | 输入 | 输出 |
|------|--------|------|------|
| 1. 分词 | `NLPProcessor.segment()` | `"人工智能是计算机..."` | `["人工智能", "计算机", "科学", ...]` |
| 2. 提取关键词 | `NLPProcessor.extract_keywords_textrank()` | 原始文本 | `[("人工智能", 0.9), ("机器学习", 0.7), ...]` |
| 3. 构建共现矩阵 | `NLPProcessor.build_cooccurrence_matrix()` | 分词结果 | `{("人工智能","机器学习"): 3, ...}` |
| 4. 构建图 | `KnowledgeGraphBuilder.build_from_cooccurrence()` | 关键词 + 共现矩阵 | NetworkX 图对象 |
| 5. 生成 HTML | `KnowledgeGraphBuilder.generate_echarts_html()` | 图对象 | 完整的 HTML 字符串 |
| 6. 显示 | `st.components.v1.html()` | HTML 字符串 | 浏览器中的交互式图谱 |

---

## 4. 逐文件深度解析

### 4.1 `app.py` — 用户界面

**一句话**：这是用户看到的网页，负责画按钮、文本框、图谱展示区。

**文件结构**（按代码顺序）：

```
┌────────────────────────────────────┐
│ 第 1-6 行   导入依赖               │
│ 第 11 行     st.set_page_config()  │  ← 必须是第一个 Streamlit 命令
│ 第 20-27 行  session_state 初始化   │  ← 给"记忆变量"设默认值
│ 第 32-72 行  process_notes() 函数  │  ← 🔥 核心逻辑：串联所有模块
│ 第 75-85 行  export_graph_html()   │  ← 导出按钮的逻辑
│ 第 91-229 行 UI 渲染               │  ← 画标题、侧边栏、输入区、展示区
└────────────────────────────────────┘
```

#### 🔑 关键概念：`st.session_state`

这是 Streamlit 的"记忆机制"。普通变量在每次点击按钮后都会重置，但 `session_state` 里的值会一直保留。你可以把它理解为"浏览器的临时数据库"。

```python
# 初始化（只会在第一次访问页面时执行）
if "graph_html" not in st.session_state:
    st.session_state.graph_html = None   # 图谱的 HTML 代码
if "keywords" not in st.session_state:
    st.session_state.keywords = []       # 关键词列表
if "summary" not in st.session_state:
    st.session_state.summary = None      # AI 摘要文本
if "has_result" not in st.session_state:
    st.session_state.has_result = False  # 是否有分析结果
```

#### 🔥 核心函数：`process_notes()`

这是整个应用的大脑，被所有 4 种输入模式调用：

```python
def process_notes(text, topk_keywords, edge_threshold, enable_llm, api_key):
    # ① 检查输入是否为空
    # ② 创建 NLPProcessor，做分词 + 关键词提取
    # ③ 如果 TextRank 提取不到关键词（文本太短），自动回退到 TF-IDF
    # ④ 构建词共现矩阵
    # ⑤ 创建 KnowledgeGraphBuilder，构建图 → 生成 HTML
    # ⑥ 把结果存入 session_state（触发页面刷新）
    # ⑦ 如果用户开启了 AI 摘要，调用 LLMProcessor
```

#### 🎛️ 四种输入模式

| 模式 | 用户操作 | 代码行 | 特殊处理 |
|------|----------|--------|----------|
| ✍️ 文本输入 | 在文本框粘贴笔记 | 141-149 | 最直接，无额外处理 |
| 📄 Markdown 编辑 | 用 Markdown 语法写笔记 | 152-159 | 预填了模板内容 |
| 🖼️ 图片上传 | 上传 jpg/png 图片 | 162-177 | 先 OCR 识别文字，再处理 |
| 📁 批量导入 | 上传多个 txt/md 文件 | 180-198 | 合并所有文件内容，统一处理 |

#### 📐 页面布局

```
┌──────────────────────────────────────────────┐
│  📚 KnowNote - 智能笔记助手                   │
├────────────┬─────────────────────────────────┤
│  侧边栏     │  主区域                          │
│            │                                 │
│  模式选择   │  ┌──────────┐  ┌─────────────┐ │
│  ● 文本     │  │ 笔记输入区 │  │  知识图谱    │ │
│  ○ Markdown│  │          │  │             │ │
│  ○ OCR     │  │ 文本框    │  │  力导向图    │ │
│  ○ 批量    │  │          │  │  (可拖拽)   │ │
│            │  │ [生成按钮]│  │             │ │
│  图谱参数   │  └──────────┘  └─────────────┘ │
│  关键词数   │                                 │
│  关系阈值   │  ┌──────────┐  ┌─────────────┐ │
│            │  │ AI 摘要   │  │  关键词标签  │ │
│  AI 增强   │  │ (可折叠)  │  │  统计信息    │ │
│  □ 启用    │  └──────────┘  └─────────────┘ │
│            │                                 │
│  导出 HTML │                                 │
├────────────┴─────────────────────────────────┤
```

---

### 4.2 `modules/nlp_processor.py` — NLP 处理器

**一句话**：把一段中文文本变成结构化的"词"和"关系"。

这是整个流程的第一步，依赖 **jieba**（结巴分词）库。类名叫 `NLPProcessor`。

#### 方法一览

```
NLPProcessor
├── __init__()                        # 初始化：加载停用词表
├── _load_stopwords()                 # 定义哪些词要忽略（的、了、是...）
├── segment(text)                     # 🔤 中文分词
├── extract_keywords_textrank(text)   # 🏷️ 关键词提取（主力算法）
├── extract_keywords_tfidf(text)      # 🏷️ 关键词提取（备用算法）
├── extract_entities(text)            # 👤 命名实体识别（人名/地名/机构）
└── build_cooccurrence_matrix(words)  # 🔗 构建词共现矩阵
```

#### ① `segment()` — 分词

```python
# 输入
text = "人工智能是计算机科学的一个分支"

# 处理过程
# 1. 正则清洗：去掉标点符号，只保留中文、英文、数字
# 2. jieba.lcut() 精确模式分词
# 3. 过滤停用词（的、了、是、在...）和单字词

# 输出
["人工智能", "计算机", "科学", "分支"]
```

> **为什么过滤单字词？** "的"、"了"、"是"这些词出现频率最高，但对理解内容毫无帮助。单字词（如"一"、"个"）通常也不是有效关键词。

#### ② `extract_keywords_textrank()` — TextRank 关键词提取

**TextRank 算法原理**（白话版）：

```
想象你在开一个投票会：
- 每个词都是一个"候选人"
- 如果两个词经常在同一个窗口（5个词范围内）出现，它们就互相"投票"
- 被很多重要词投票的词，自己也变得重要
- 最后按得票数排序，取前 N 个
```

这比简单的"数出现次数"（TF-IDF）更智能，因为它考虑了**词与词之间的关系**。

```python
keywords = nlp.extract_keywords_textrank("人工智能...", topK=10)
# 返回: [("人工智能", 0.92), ("机器学习", 0.78), ("深度学习", 0.65), ...]
#       每个元素是 (词, 权重)，权重越高越重要
```

#### ③ `extract_keywords_tfidf()` — TF-IDF 备用算法

当文本太短、TextRank 无法有效提取时，回退到 TF-IDF（词频-逆文档频率）。这是一个更基础但永远能出结果的算法。

#### ④ `extract_entities()` — 命名实体识别

利用 jieba 的词性标注功能，识别出：

| 词性标记 | 含义 | 示例 |
|----------|------|------|
| `nr` | 人名 | 张三、李白 |
| `ns` | 地名 | 北京、上海 |
| `nt` | 机构名 | 清华大学、微软 |
| `nz` | 专有名词 | Transformer、区块链 |

```python
entities = nlp.extract_entities("乔布斯创立了苹果公司")
# 返回: [
#   {"word": "乔布斯", "type": "人名", "pos": "nr"},
#   {"word": "苹果公司", "type": "机构名", "pos": "nt"}
# ]
```

#### ⑤ `build_cooccurrence_matrix()` — 共现矩阵（最关键的概念）

这是构建知识图谱的**核心数据来源**。原理非常简单：

```
假设分词结果是：["人工智能", "计算机", "科学", "人工智能", "机器学习"]
窗口大小 = 2

滑动窗口演示：
┌─────────────────────┐
│ 人工智能  计算机  科学 │  → 记录: (人工智能,计算机)=1, (人工智能,科学)=1, (计算机,科学)=1
    └─────────────────────┘
     计算机  科学  人工智能  → 记录: (计算机,科学)=2, (计算机,人工智能)=2, (科学,人工智能)=2
         └─────────────────────┘
          科学  人工智能  机器学习 → 记录: (科学,人工智能)=3, (科学,机器学习)=1, (人工智能,机器学习)=1

最终共现矩阵：
{
    ("人工智能", "计算机"): 2,
    ("人工智能", "科学"):   3,
    ("人工智能", "机器学习"): 1,
    ("计算机", "科学"):     2,
    ("科学", "机器学习"):   1,
}
```

> 两个词在窗口内同时出现的次数越多，它们的关系就越紧密。这个数字就是知识图谱中"边的权重"。

---

### 4.3 `modules/graph_builder.py` — 图谱构建器

**一句话**：把"一堆词和它们的关系"变成"可以在浏览器里拖拽的图"。

依赖 **NetworkX**（图论计算库）和 **ECharts**（前端图表库）。

#### 核心概念：什么是"图"

```
图 = 节点 + 边

节点（Node）= 关键词，如"人工智能"
边（Edge）= 两个词的关系，如"人工智能"和"机器学习"之间有一条线
边的权重 = 关系的强度，线越粗表示关联越强
```

#### 方法一览

```
KnowledgeGraphBuilder
├── __init__()                         # 初始化空图
├── _get_node_id(name)                 # 给每个节点分配唯一 ID
├── build_from_cooccurrence(...)       # 🏗️ 从共现矩阵构建图
├── build_from_entities(...)           # 🏗️ 从命名实体构建图（备用）
├── to_echarts_format()                # 🔄 转换为 ECharts 数据格式
└── generate_echarts_html()            # 🎨 生成完整 HTML 页面
```

#### ① `build_from_cooccurrence()` — 从共现到图

```python
# 输入
keywords = [("人工智能", 0.92), ("机器学习", 0.78), ...]
cooccurrence = {("人工智能","机器学习"): 3, ("人工智能","深度学习"): 2, ...}

# 处理
# 1. 清空旧图
# 2. 把每个关键词加为节点，权重映射为节点大小
# 3. 遍历共现矩阵，归一化权重后加边
# 4. 过滤：权重低于 edge_threshold 的边不显示

# 结果
graph = nx.Graph()
graph.nodes: {"node_1": {"name": "人工智能", "size": 46}, ...}
graph.edges:  {("node_1","node_2"): {"weight": 1.0}, ...}
```

#### ② `to_echarts_format()` — 格式转换

NetworkX 的图对象不能直接给 ECharts 用，需要转换格式：

```python
# NetworkX 内部格式
graph.nodes["node_1"]  # {"name": "人工智能", "size": 46, ...}

# ECharts 需要的格式
{
    "categories": [{"name": "概念"}, {"name": "人名"}, ...],  # 分类图例
    "nodes": [
        {"name": "人工智能", "category": 0, "symbolSize": 46},  # 节点列表
        ...
    ],
    "links": [
        {"source": "人工智能", "target": "机器学习", "value": 1.0},  # 边列表
        ...
    ]
}
```

#### ③ `generate_echarts_html()` — 生成可视化页面

这个函数返回一个**完整的、可以独立保存的 HTML 文件**。它内嵌了：

- ECharts CDN 引用（从 jsdelivr 加载）
- 图谱数据（以 JSON 格式硬编码在 JS 中）
- 力导向布局配置

**力导向布局是什么？**

```
想象每个节点都是一个带磁力的小球：
- 斥力（repulsion）：每个节点互相排斥，不会重叠
- 引力（edges pull）：有连线的节点互相吸引
- 重力（gravity）：防止节点飞出屏幕
- 摩擦力（friction）：让运动逐渐停下来

最终所有节点会稳定在一个"看起来舒服"的位置
```

ECharts 配置的关键参数：

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `repulsion: 500` | 节点斥力 | 越大节点越分散 |
| `edgeLength: [50, 100]` | 边长范围 | 控制有关系的节点距离 |
| `gravity: 0.1` | 重力因子 | 越大节点越向中心靠拢 |
| `friction: 0.6` | 摩擦力 | 越大动画越快停止 |
| `roam: true` | 允许缩放/拖拽 | 用户可以用鼠标操作 |
| `draggable: true` | 节点可拖拽 | 用户可以把节点拖到任意位置 |

---

### 4.4 `modules/ocr_processor.py` — OCR 处理器

**一句话**：把图片里的文字"读"出来。

依赖 **PaddleOCR**（百度开源的 OCR 引擎）。

#### 方法一览

```
OCRProcessor
├── __init__()            # 创建临时目录
├── init_engine()         # 🚀 懒加载 OCR 引擎
└── extract_text(file)    # 👁️ 从图片中提取文字
```

#### ① `init_engine()` — 懒加载

```python
def init_engine(self):
    if self.ocr is None:        # 只在第一次调用时才加载
        from paddleocr import PaddleOCR
        self.ocr = PaddleOCR(
            use_angle_cls=True,  # 自动纠正文字方向
            lang="ch",           # 中英文混合识别
            show_log=False       # 不打印调试日志
        )
    return self.ocr
```

> **为什么懒加载？** PaddleOCR 模型文件很大（几百 MB），如果一启动就加载会占用大量内存。懒加载让 OCR 引擎只在用户真正上传图片时才初始化。

#### ② `extract_text()` — 识别文字

```python
# 处理流程
# 1. 初始化引擎（首次调用）
# 2. 生成唯一文件名（避免多用户冲突）
# 3. 将上传的文件保存到临时目录
# 4. 调用 PaddleOCR 识别
# 5. 过滤低置信度结果（confidence < 0.5 的丢弃）
# 6. 返回识别出的文本
# 7. finally 块清理临时文件

# PaddleOCR 返回格式
# [
#   [  # 第一页
#       ([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], ("识别的文字", 0.98)),  # 0.98 是置信度
#       ([[...]], ("第二行文字", 0.87)),
#   ]
# ]
```

---

### 4.5 `modules/llm_processor.py` — 大模型处理器

**一句话**：调用大语言模型（GPT / DeepSeek / 文心一言）来帮你总结笔记。

依赖 **requests**（HTTP 请求库），不依赖任何 AI SDK。

#### 方法一览

```
LLMProcessor
├── __init__(api_key, api_type)            # 选择平台（openai/deepseek/baidu）
├── summarize(text)                        # 📝 生成笔记摘要
├── ask_question(text, question)           # ❓ 基于笔记回答问题
├── extract_keywords_with_llm(text)       # 🏷️ LLM 提取关键词
├── _call_llm(prompt)                     # 📞 实际发送 HTTP 请求
├── _build_headers()                      # 🔑 构建鉴权头
├── _build_payload(config, prompt)        # 📦 构建请求体
└── _parse_response(result)              # 📥 解析响应
```

#### 核心设计：策略模式

```python
API_CONFIG = {
    "openai":   {"url": "https://api.openai.com/v1/chat/completions",  "model": "gpt-3.5-turbo"},
    "deepseek": {"url": "https://api.deepseek.com/v1/chat/completions", "model": "deepseek-chat"},
    "baidu":    {"url": "https://aip.baidubce.com/...",                "model": "ernie-speed-128k"},
}
```

创建时指定类型，调用时自动路由：

```python
# 用 DeepSeek（最便宜）
llm = LLMProcessor(api_key="sk-xxx", api_type="deepseek")
summary = llm.summarize("很长的笔记...")

# 用百度（有免费额度）
llm = LLMProcessor(api_key="your_token", api_type="baidu")
summary = llm.summarize("很长的笔记...")
```

#### API 调用流程

```
summarize("笔记内容")
    │
    ▼
构建 Prompt（提示词模板）
    │
    ▼
_call_llm(prompt)
    │
    ├── _build_headers()     → {"Authorization": "Bearer sk-xxx"}
    ├── _build_payload()     → {"model": "deepseek-chat", "messages": [...]}
    │
    ├── requests.post(url, headers, json, timeout=60)
    │
    ├── 异常处理（超时/HTTP错误/网络错误）
    │
    └── _parse_response()   → 从 JSON 中提取出文本内容
```

#### 错误处理分层

```python
except requests.exceptions.Timeout:        # 网络超时 → 提示稍后重试
except requests.exceptions.HTTPError:      # 4xx/5xx → 显示状态码和详情
except requests.exceptions.ConnectionError:# 无法连接 → 提示检查网络
except Exception:                          # 未知错误 → 显示异常类型
```

---

### 4.6 `modules/__init__.py` — 模块入口

**一句话**：让 `modules/` 目录变成一个"包"，并声明对外提供什么。

```python
from .nlp_processor import NLPProcessor
from .graph_builder import KnowledgeGraphBuilder
from .ocr_processor import OCRProcessor
from .llm_processor import LLMProcessor

__all__ = ["NLPProcessor", "KnowledgeGraphBuilder", "OCRProcessor", "LLMProcessor"]
```

有了这个文件，`app.py` 就可以用简洁的方式导入：

```python
# ✅ 可以这样写（因为有 __init__.py）
from modules.nlp_processor import NLPProcessor

# ❌ 如果没有 __init__.py，这行会报错
```

> `__all__` 列表定义了 `from modules import *` 时会导出哪些名字。它是一个"公开 API 声明"。

---

### 4.7 `requirements.txt` — 依赖清单

```txt
streamlit>=1.28.0    # Web 界面框架
jieba>=0.42.1        # 中文分词
networkx>=3.1        # 图结构与算法
pandas>=2.0.0        # 数据处理
paddleocr>=2.7.0     # OCR 文字识别（可选）
paddlepaddle>=2.5.0  # PaddleOCR 的深度学习引擎（可选）
requests>=2.31.0     # HTTP 请求（调用 AI API）
```

> `>=` 表示"不低于这个版本"，pip 会自动安装符合条件的最新版。

---

## 5. 如何运行

### 第一步：安装依赖

```bash
# 基础运行（不含 OCR）
pip install streamlit jieba networkx pandas requests

# 完整安装（含 OCR）
pip install -r requirements.txt
```

### 第二步：启动

```bash
streamlit run app.py
```

浏览器自动打开 `http://localhost:8501`。

### 第三步：使用

1. 在左侧文本区粘贴一段中文笔记
2. 点击「🚀 生成知识图谱」
3. 右侧出现可拖拽的知识图谱
4. （可选）勾选「启用 AI 摘要」，输入 API Key，获得智能摘要

---

## 6. 如何扩展

### 想加新的输入方式？

在 `app.py` 侧边栏的 `mode` 列表加一项，然后在主区域加一个 `elif` 分支。

### 想支持新的 AI 平台？

在 `llm_processor.py` 的 `API_CONFIG` 字典加一条配置，不需要改其他任何代码。

### 想换一种图谱样式？

修改 `graph_builder.py` 中 `generate_echarts_html()` 的 ECharts 配置项（如 `repulsion`、`edgeLength`），或者换一种布局算法（如 `circular` 环形布局）。

### 想添加新的 NLP 功能？

在 `nlp_processor.py` 中加新方法，然后在 `process_notes()` 中调用它。

### 模块可以在其他项目中复用吗？

可以。`modules/` 下的所有模块都不依赖 Streamlit，可以直接复制到其他 Python 项目中使用。

---

> 📅 最后更新：2026-06-10
