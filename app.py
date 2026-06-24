# app.py - 主程序入口
import streamlit as st
from modules.nlp_processor import NLPProcessor
from modules.graph_builder import KnowledgeGraphBuilder
from modules.ocr_processor import OCRProcessor
from modules.llm_processor import LLMProcessor

# ============================================================
# 页面配置 — 必须是第一个 Streamlit 命令
# ============================================================
st.set_page_config(
    page_title="KnowNote - 智能笔记助手",
    page_icon="📚",
    layout="wide"
)

# ============================================================
# Session State 初始化
# ============================================================
if "graph_html" not in st.session_state:
    st.session_state.graph_html = None
if "keywords" not in st.session_state:
    st.session_state.keywords = []
if "summary" not in st.session_state:
    st.session_state.summary = None
if "has_result" not in st.session_state:
    st.session_state.has_result = False
if "user_input_text" not in st.session_state:
    st.session_state.user_input_text = ""
if "keyword_info" not in st.session_state:
    st.session_state.keyword_info = None  # LLM 返回的关键词详解
if "keyword_info_loading" not in st.session_state:
    st.session_state.keyword_info_loading = False
if "saved_api_key" not in st.session_state:
    st.session_state.saved_api_key = ""
if "saved_api_type" not in st.session_state:
    st.session_state.saved_api_type = "deepseek"


# ============================================================
# 核心处理函数
# ============================================================
def process_notes(text, topk_keywords, edge_threshold, enable_llm, api_key, api_type):
    """
    处理用户输入的笔记文本，生成知识图谱
    """
    if not text or not text.strip():
        st.warning("请输入笔记内容")
        return

    # 保存用户输入到 session_state（用于后续关键词查询时提供上下文）
    st.session_state.user_input_text = text

    with st.spinner("正在分析笔记内容..."):
        # 1. NLP 处理：分词 + 关键词提取 + 共现矩阵
        nlp = NLPProcessor()
        words = nlp.segment(text)
        keywords = nlp.extract_keywords_textrank(text, topK=topk_keywords)

        # TextRank 可能返回空（文本太短时），回退到 TF-IDF
        if not keywords:
            keywords = nlp.extract_keywords_tfidf(text, topK=topk_keywords)

        cooccurrence = nlp.build_cooccurrence_matrix(words)

        # 2. 构建知识图谱
        builder = KnowledgeGraphBuilder()
        builder.build_from_cooccurrence(keywords, cooccurrence, edge_threshold)

        # 3. 生成 ECharts HTML 并存入 session_state
        st.session_state.graph_html = builder.generate_echarts_html()
        st.session_state.keywords = [kw for kw, _ in keywords]
        st.session_state.has_result = True
        st.session_state.keyword_info = None  # 重置关键词查询结果

        # 4. 可选：AI 摘要
        if enable_llm:
            if not api_key:
                st.warning("请在侧边栏输入 API Key 以启用 AI 摘要")
            else:
                with st.spinner("正在生成 AI 摘要..."):
                    try:
                        llm = LLMProcessor(api_key, api_type=api_type)
                        result = llm.summarize(text)
                        st.session_state.summary = result
                    except Exception as e:
                        st.session_state.summary = {
                            "success": False,
                            "content": f"AI 摘要生成异常: {str(e)}"
                        }


def query_keyword_info(keyword, api_key, api_type):
    """查询单个关键词的 AI 详解"""
    if not api_key:
        return {"success": False, "content": "⚠️ 请先在侧边栏输入 API Key"}

    try:
        llm = LLMProcessor(api_key, api_type=api_type)
        context = st.session_state.get("user_input_text", "")
        return llm.get_keyword_info(keyword, context=context)
    except Exception as e:
        return {"success": False, "content": f"❌ 查询异常: {str(e)}"}


def export_graph_as_html():
    """将当前图谱导出为独立的 HTML 文件"""
    if "graph_html" in st.session_state and st.session_state.graph_html:
        import base64
        html_bytes = st.session_state.graph_html.encode("utf-8")
        b64 = base64.b64encode(html_bytes).decode()
        href = f'<a href="data:text/html;base64,{b64}" download="knowledge_graph.html">📥 下载 HTML 文件</a>'
        st.markdown(href, unsafe_allow_html=True)
        st.success("点击上方链接即可下载")
    else:
        st.warning("请先生成知识图谱")


# ============================================================
# 检查是否有来自图谱节点点击的关键词
# ============================================================
clicked_keyword = st.query_params.get("keyword")

# ============================================================
# 页面标题
# ============================================================
st.title("📚 KnowNote - 智能笔记助手")
st.markdown("将零散笔记转化为知识图谱，让学习更高效")

# ============================================================
# 侧边栏 - 功能设置
# ============================================================
with st.sidebar:
    st.header("⚙️ 功能设置")

    # 模式选择
    mode = st.radio(
        "选择输入方式",
        ["✍️ 文本输入", "📄 Markdown 编辑", "🖼️ 图片上传(OCR)", "📁 批量导入"],
        key="input_mode"
    )

    st.divider()

    # 图谱构建参数
    st.subheader("图谱参数")
    topk_keywords = st.slider("关键词数量", 5, 20, 10)
    edge_threshold = st.slider("关系强度阈值", 0.1, 1.0, 0.3)

    st.divider()

    # 高级功能开关
    st.subheader("🤖 AI 增强")
    enable_llm = st.checkbox("启用 AI 摘要（需 API Key）")
    api_key = st.session_state.saved_api_key
    api_type = st.session_state.saved_api_type
    if enable_llm:
        api_type = st.selectbox(
            "API 平台",
            ["deepseek", "openai", "baidu"],
            index=["deepseek", "openai", "baidu"].index(st.session_state.saved_api_type)
                if st.session_state.saved_api_type in ["deepseek", "openai", "baidu"]
                else 0,
            format_func=lambda x: {
                "deepseek": "DeepSeek（推荐，国内可用，极低价格）",
                "openai": "OpenAI（需代理）",
                "baidu": "百度文心（有免费额度）",
            }[x],
            help="DeepSeek 国内可直接访问；OpenAI 需要科学上网；百度文心有免费额度"
        )
        placeholder_map = {
            "deepseek": "sk-...  (在 platform.deepseek.com 获取)",
            "openai": "sk-...  (在 platform.openai.com 获取)",
            "baidu": "Access Token (在 console.bce.baidu.com 获取)",
        }
        api_key = st.text_input(
            "API Key",
            type="password",
            value=st.session_state.saved_api_key,
            placeholder=placeholder_map.get(api_type, "请输入 API Key"),
        )
        # 持久化保存 API Key（避免页面刷新丢失）
        st.session_state.saved_api_key = api_key
        st.session_state.saved_api_type = api_type

    st.divider()

    # 导出选项
    st.subheader("📤 导出")
    col_export1, col_export2 = st.columns(2)
    with col_export1:
        if st.button("导出 HTML", use_container_width=True):
            export_graph_as_html()

# ============================================================
# 主区域 - 两列布局
# ============================================================
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📝 笔记输入区")

    # --- 模式1: 文本输入 ---
    if mode == "✍️ 文本输入":
        user_input = st.text_area(
            "粘贴你的笔记：",
            height=300,
            placeholder="在此输入或粘贴你的笔记内容...\n\n例如：\n人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。机器学习是人工智能的核心领域之一。"
        )

        if st.button("🚀 生成知识图谱", type="primary", use_container_width=True):
            process_notes(user_input, topk_keywords, edge_threshold, enable_llm, api_key, api_type)
            st.rerun()

    # --- 模式2: Markdown 编辑 ---
    elif mode == "📄 Markdown 编辑":
        md_input = st.text_area(
            "Markdown 格式笔记：",
            height=300,
            value="# 笔记标题\n\n## 第一节\n\n这里是笔记内容..."
        )
        if st.button("🚀 生成知识图谱", type="primary", use_container_width=True):
            process_notes(md_input, topk_keywords, edge_threshold, enable_llm, api_key, api_type)
            st.rerun()

    # --- 模式3: 图片上传 OCR ---
    elif mode == "🖼️ 图片上传(OCR)":
        uploaded_file = st.file_uploader(
            "上传包含文字的图片",
            type=["jpg", "png", "jpeg", "bmp", "webp"]
        )
        if uploaded_file and st.button("🚀 OCR 识别并生成图谱", type="primary", use_container_width=True):
            with st.spinner("正在进行 OCR 识别..."):
                ocr = OCRProcessor()
                text = ocr.extract_text(uploaded_file)
                if text and not text.startswith("OCR"):
                    st.success("OCR 识别完成！")
                    with st.expander("📋 查看识别结果"):
                        st.write(text)
                    process_notes(text, topk_keywords, edge_threshold, enable_llm, api_key, api_type)
                    st.rerun()
                else:
                    st.error(f"OCR 识别失败: {text}")

    # --- 模式4: 批量导入 ---
    elif mode == "📁 批量导入":
        uploaded_files = st.file_uploader(
            "选择多个笔记文件（支持 .txt / .md）",
            type=["txt", "md"],
            accept_multiple_files=True
        )
        if uploaded_files:
            st.info(f"已选择 {len(uploaded_files)} 个文件")
            all_text = []
            for f in uploaded_files:
                content = f.read().decode("utf-8", errors="ignore")
                all_text.append(f"## {f.name}\n\n{content}")
            combined = "\n\n---\n\n".join(all_text)

            with st.expander(f"📋 预览合并内容（{len(uploaded_files)} 个文件）"):
                st.text_area("合并内容", combined, height=200, disabled=True)

            if st.button("🚀 批量生成知识图谱", type="primary", use_container_width=True):
                process_notes(combined, topk_keywords, edge_threshold, enable_llm, api_key, api_type)
                st.rerun()

# ============================================================
# 右栏 - 知识图谱展示
# ============================================================
with col2:
    st.subheader("🗺️ 知识图谱")

    if st.session_state.has_result and st.session_state.graph_html:
        st.components.v1.html(st.session_state.graph_html, height=580)
    else:
        st.info("👈 在左侧输入笔记，点击「生成知识图谱」按钮")

    # ---- 关键词点击查询结果 ----
    if clicked_keyword and st.session_state.has_result:
        # 去重：如果已经查询过同一个关键词，不再重复查询
        last_queried = st.session_state.get("_last_queried_keyword", "")
        if clicked_keyword != last_queried or st.session_state.get("keyword_info") is None:
            st.session_state._last_queried_keyword = clicked_keyword
            with st.spinner(f"正在查询「{clicked_keyword}」的 AI 详解..."):
                result = query_keyword_info(clicked_keyword, api_key, api_type)
                st.session_state.keyword_info = result

        # 显示关键词查询结果
        if st.session_state.keyword_info:
            result = st.session_state.keyword_info
            if result.get("success"):
                st.success(f"✅ 「**{clicked_keyword}**」AI 详解已生成")
                with st.expander(f"🤖 「{clicked_keyword}」详解", expanded=True):
                    st.markdown(result["content"])
                    # 清除按钮
                    if st.button("❌ 关闭详解", key="clear_keyword_info"):
                        st.session_state.keyword_info = None
                        st.session_state._last_queried_keyword = ""
                        st.query_params.clear()
                        st.rerun()
            else:
                st.error(result.get("content", "查询失败"))

    # ---- AI 摘要展示 ----
    if st.session_state.summary:
        summary = st.session_state.summary
        if isinstance(summary, dict):
            if summary.get("success"):
                with st.expander("🤖 AI 摘要", expanded=True):
                    st.write(summary["content"])
        else:
            # 兼容旧的字符串格式
            with st.expander("🤖 AI 摘要", expanded=True):
                st.write(summary)

    # ---- 关键词标签 ----
    if st.session_state.keywords:
        st.subheader("🏷️ 核心关键词")
        st.caption("点击图谱节点或下方按钮可查询 AI 详解")
        cols = st.columns(5)
        for i, kw in enumerate(st.session_state.keywords[:15]):
            with cols[i % 5]:
                # 用 link_button 模拟，点击后设置 query_param
                if st.button(kw, key=f"kw_{i}", use_container_width=True,
                             help=f"点击查询「{kw}」的 AI 详解"):
                    st.query_params["keyword"] = kw
                    st.rerun()

    # ---- 图谱统计 ----
    if st.session_state.has_result:
        with st.expander("📊 图谱统计"):
            kw_count = len(st.session_state.keywords)
            st.metric("关键词节点数", kw_count)
            st.caption("关系强度阈值: " + str(edge_threshold))
