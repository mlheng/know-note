# app.py - 主程序入口
import streamlit as st
from modules.nlp_processor import NLPProcessor
from modules.graph_builder import KnowledgeGraphBuilder
from modules.llm_processor import LLMProcessor
from modules.mindmap_builder import MindMapBuilder
from modules.ocr_processor import OCRProcessor

# ============================================================
# 页面配置 — 必须是第一个 Streamlit 命令
# ============================================================
st.set_page_config(
    page_title="KnowNote - 智能笔记助手",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 自定义 CSS 美化
# ============================================================
st.markdown("""
<style>
    /* ---- 全局 ----
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #e4e9f0 100%); }
    .main .block-container { padding-top: 1rem; }

    /* ---- 标题区 ----
    .app-title { font-size: 2.2rem; font-weight: 800; margin-bottom: 0; color: #1a1a2e; }
    .app-subtitle { color: #666; font-size: 0.95rem; margin-top: -0.3rem; margin-bottom: 1rem; }

    /* ---- 输入区卡片 ----
    .input-card {
        background: #fff; border-radius: 16px; padding: 1.5rem 1.8rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06); margin-bottom: 1rem;
    }

    /* ---- 结果区卡片 ----
    .result-header {
        display: flex; align-items: center; justify-content: space-between;
        flex-wrap: wrap; gap: 0.5rem;
    }
    .result-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #fff; padding: 0.25rem 0.75rem; border-radius: 20px;
        font-size: 0.8rem; font-weight: 600;
    }

    /* ---- 统计卡片 ----
    .stat-row {
        display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0;
    }
    .stat-card {
        flex: 1; min-width: 100px; background: #fff; border-radius: 12px;
        padding: 1rem 1.2rem; text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        border-left: 4px solid #667eea;
    }
    .stat-card .stat-num {
        font-size: 2rem; font-weight: 700; color: #1a1a2e; line-height: 1.2;
    }
    .stat-card .stat-label { font-size: 0.8rem; color: #888; margin-top: 0.2rem; }

    /* ---- 关键词标签 ----
    .kw-container { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.5rem 0 1rem; }

    /* ---- 页脚 ----
    .app-footer { text-align: center; color: #aaa; font-size: 0.75rem; margin-top: 2rem; padding: 1rem 0; }

    /* ---- expander 样式 ----
    [data-testid="stExpander"] { background: #fff; border-radius: 12px; box-shadow: 0 1px 6px rgba(0,0,0,0.04); }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Session State 初始化
# ============================================================
defaults = {
    "graph_html": None,
    "keywords": [],
    "summary": None,
    "has_result": False,
    "user_input_text": "",
    "keyword_info": None,
    "keyword_info_loading": False,
    "saved_api_key": "",
    "saved_api_type": "deepseek",
    "mindmap_html": None,
    "view_mode": "🗺️ 知识图谱",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================
# 核心处理函数
# ============================================================
def process_notes(text, topk_keywords, edge_threshold, enable_llm, api_key, api_type):
    """处理用户输入的笔记文本，生成知识图谱"""
    if not text or not text.strip():
        st.warning("请输入笔记内容")
        return

    st.session_state.user_input_text = text

    with st.spinner("正在分析笔记内容..."):
        nlp = NLPProcessor()
        words = nlp.segment(text)
        keywords = nlp.extract_keywords_textrank(text, topK=topk_keywords)
        if not keywords:
            keywords = nlp.extract_keywords_tfidf(text, topK=topk_keywords)
        cooccurrence = nlp.build_cooccurrence_matrix(words)

        builder = KnowledgeGraphBuilder()
        builder.build_from_cooccurrence(keywords, cooccurrence, edge_threshold)

        st.session_state.graph_html = builder.generate_echarts_html()
        st.session_state.keywords = [kw for kw, _ in keywords]
        st.session_state.has_result = True
        st.session_state.keyword_info = None

        # NLP 思维导图
        builder_mm = MindMapBuilder()
        nlp_tree = builder_mm.build_tree_from_graph(builder.graph, keywords)
        st.session_state.mindmap_html = builder_mm.generate_echarts_tree_html(nlp_tree)

        # AI 增强
        if enable_llm:
            if not api_key:
                st.warning("请在侧边栏输入 API Key 以启用 AI 增强")
            else:
                with st.spinner("正在生成 AI 摘要和思维导图..."):
                    try:
                        llm = LLMProcessor(api_key, api_type=api_type)
                        st.session_state.summary = llm.summarize(text)
                        mm_result = llm.generate_mindmap(text)
                        if mm_result.get("success"):
                            st.session_state.mindmap_html = builder_mm.generate_markmap_html(
                                mm_result["content"])
                    except Exception as e:
                        st.session_state.summary = {
                            "success": False,
                            "content": f"AI 增强异常: {str(e)}"}


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
    if "graph_html" in st.session_state and st.session_state.graph_html:
        import base64
        b64 = base64.b64encode(st.session_state.graph_html.encode("utf-8")).decode()
        href = f'<a href="data:text/html;base64,{b64}" download="knowledge_graph.html">📥 下载知识图谱</a>'
        st.markdown(href, unsafe_allow_html=True)
    else:
        st.warning("请先生成知识图谱")


def export_mindmap_as_html():
    if "mindmap_html" in st.session_state and st.session_state.mindmap_html:
        import base64
        b64 = base64.b64encode(st.session_state.mindmap_html.encode("utf-8")).decode()
        href = f'<a href="data:text/html;base64,{b64}" download="mindmap.html">📥 下载思维导图</a>'
        st.markdown(href, unsafe_allow_html=True)
    else:
        st.warning("请先生成思维导图")


def _render_input_area(mode, topk_keywords, edge_threshold, enable_llm, api_key, api_type):
    """渲染笔记输入区域（根据 mode 切换不同输入方式）"""
    if mode == "✍️ 文本输入":
        text = st.text_area(
            "粘贴你的笔记：", height=280,
            placeholder="在此输入或粘贴你的笔记内容...\n\n例如：\n人工智能是计算机科学的一个分支...机器学习是人工智能的核心领域之一。深度学习是机器学习的重要方法。")
        if st.button("🚀 生成知识图谱", type="primary", use_container_width=True):
            process_notes(text, topk_keywords, edge_threshold, enable_llm, api_key, api_type)
            st.rerun()

    elif mode == "📄 Markdown 编辑":
        text = st.text_area("Markdown 格式笔记：", height=280,
                            value="# 笔记标题\n\n## 第一节\n\n这里是笔记内容...")
        if st.button("🚀 生成知识图谱", type="primary", use_container_width=True):
            process_notes(text, topk_keywords, edge_threshold, enable_llm, api_key, api_type)
            st.rerun()

    elif mode == "🖼️ 图片上传(OCR)":
        uploaded_file = st.file_uploader("上传包含文字的图片", type=["jpg", "png", "jpeg", "bmp", "webp"])
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
                    st.error(text)

    elif mode == "📁 批量导入":
        files = st.file_uploader("选择多个笔记文件（.txt / .md）", type=["txt", "md"],
                                 accept_multiple_files=True)
        if files:
            st.info(f"已选择 {len(files)} 个文件")
            all_text = []
            for f in files:
                content = f.read().decode("utf-8", errors="ignore")
                all_text.append(f"## {f.name}\n\n{content}")
            combined = "\n\n---\n\n".join(all_text)
            with st.expander(f"📋 预览合并内容（{len(files)} 个文件）"):
                st.text_area("合并内容", combined, height=200, disabled=True)
            if st.button("🚀 批量生成知识图谱", type="primary", use_container_width=True):
                process_notes(combined, topk_keywords, edge_threshold, enable_llm, api_key, api_type)
                st.rerun()


# ============================================================
# 页面标题
# ============================================================
st.markdown('<p class="app-title">📚 KnowNote · 智能笔记助手</p>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">将零散笔记转化为交互式知识图谱，让学习更高效</p>',
            unsafe_allow_html=True)

# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.header("⚙️ 功能设置")

    mode = st.radio("📥 输入方式", ["✍️ 文本输入", "📄 Markdown 编辑",
                    "🖼️ 图片上传(OCR)", "📁 批量导入"], key="input_mode")

    st.divider()
    st.subheader("📐 图谱参数")
    topk_keywords = st.slider("关键词数量", 5, 20, 10)
    edge_threshold = st.slider("关系强度阈值", 0.1, 1.0, 0.3)

    st.divider()
    st.subheader("🤖 AI 增强")
    enable_llm = st.checkbox("启用 AI 增强（摘要/详解/联想）")
    api_key = st.session_state.saved_api_key
    api_type = st.session_state.saved_api_type
    if enable_llm:
        api_type = st.selectbox("API 平台", ["deepseek", "openai", "baidu"],
                                index=["deepseek", "openai", "baidu"].index(
                                    st.session_state.saved_api_type)
                                if st.session_state.saved_api_type in [
                                    "deepseek", "openai", "baidu"] else 0,
                                format_func=lambda x: {
                                    "deepseek": "DeepSeek（推荐）", "openai": "OpenAI", "baidu": "百度文心"}[x])
        placeholder_map = {
            "deepseek": "sk-... (platform.deepseek.com)",
            "openai": "sk-... (platform.openai.com)",
            "baidu": "Access Token (console.bce.baidu.com)",
        }
        api_key = st.text_input("API Key", type="password",
                                value=st.session_state.saved_api_key,
                                placeholder=placeholder_map.get(api_type, ""))
        st.session_state.saved_api_key = api_key
        st.session_state.saved_api_type = api_type

    st.divider()
    st.subheader("📤 导出")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📥 图谱", use_container_width=True):
            export_graph_as_html()
    with c2:
        if st.button("📥 导图", use_container_width=True):
            export_mindmap_as_html()

    # 隐藏桥接按钮（用于 iframe 通信触发 rerun）
    st.button("⟳", key="__bridge_btn__")

# ============================================================
# 主区域 — 自适应布局
# ============================================================
clicked_keyword = st.query_params.get("keyword")

# ---- 输入区（有结果时折叠） ----
if st.session_state.has_result:
    with st.expander("📝 笔记输入区（点击展开修改）", expanded=False):
        _render_input_area(mode, topk_keywords, edge_threshold,
                           enable_llm, api_key, api_type)
else:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    _render_input_area(mode, topk_keywords, edge_threshold,
                       enable_llm, api_key, api_type)
    st.markdown('</div>', unsafe_allow_html=True)

# ---- 结果区（全宽） ----
if st.session_state.has_result:
    st.markdown("---")

    # 统计卡片
    kw_count = len(st.session_state.keywords)
    has_ai = st.session_state.summary is not None
    has_keyword_query = st.session_state.keyword_info is not None

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-num">{kw_count}</div>
            <div class="stat-label">🏷️ 关键词节点</div>
        </div>
        <div class="stat-card" style="border-left-color:#52c41a;">
            <div class="stat-num">{'✅' if has_ai else '—'}</div>
            <div class="stat-label">🤖 AI 摘要</div>
        </div>
        <div class="stat-card" style="border-left-color:#fa8c16;">
            <div class="stat-num">{'✅' if has_keyword_query else '—'}</div>
            <div class="stat-label">🔍 关键词查询</div>
        </div>
        <div class="stat-card" style="border-left-color:#eb2f96;">
            <div class="stat-num">{'🧠'}</div>
            <div class="stat-label">思维导图</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 视图切换 + 知识可视化
    col_viz, col_info = st.columns([2.2, 1])

    with col_viz:
        # 视图切换按钮
        tab1, tab2 = st.tabs(["🗺️ 知识图谱", "🧠 思维导图"])
        with tab1:
            if st.session_state.graph_html:
                st.components.v1.html(st.session_state.graph_html, height=620)
            else:
                st.info("图谱尚未生成")
        with tab2:
            if st.session_state.mindmap_html:
                st.components.v1.html(st.session_state.mindmap_html, height=620)
            else:
                st.info("思维导图尚未生成")

    # ---- 右侧信息面板 ----
    with col_info:
        # 关键词点击查询结果
        if clicked_keyword:
            last_queried = st.session_state.get("_last_queried_keyword", "")
            prev_info = st.session_state.get("keyword_info")
            prev_failed = prev_info is not None and not prev_info.get("success", True)
            if clicked_keyword != last_queried or prev_info is None or prev_failed:
                st.session_state._last_queried_keyword = clicked_keyword
                with st.spinner(f"🔍 查询「{clicked_keyword}」..."):
                    st.session_state.keyword_info = query_keyword_info(
                        clicked_keyword, api_key, api_type)

            if st.session_state.keyword_info:
                result = st.session_state.keyword_info
                if result.get("success"):
                    st.success(f"✅ 「**{clicked_keyword}**」")
                    st.markdown(result["content"])
                    if st.button("✕ 关闭", key="clear_kw"):
                        st.session_state.keyword_info = None
                        st.session_state._last_queried_keyword = ""
                        st.query_params.clear()
                        st.rerun()
                else:
                    st.error(result.get("content", "查询失败"))

        # AI 摘要
        if st.session_state.summary:
            summary = st.session_state.summary
            with st.expander("🤖 AI 摘要", expanded=True):
                if isinstance(summary, dict):
                    if summary.get("success"):
                        st.write(summary["content"])
                    else:
                        st.warning(summary.get("content"))
                else:
                    st.write(summary)

        # 关键词标签云
        if st.session_state.keywords:
            st.markdown("##### 🏷️ 核心关键词")
            st.caption("点击查询 AI 详解")
            cols = st.columns(3)
            for i, kw in enumerate(st.session_state.keywords[:15]):
                with cols[i % 3]:
                    if st.button(kw, key=f"kw_{i}", use_container_width=True):
                        st.query_params["keyword"] = kw
                        st.rerun()

    # ---- 底部信息 ----
    st.caption(
        f"📊 关系强度阈值: {edge_threshold} | 💡 提示：点击图谱中任意节点可查询 AI 详解")

# ---- 脚注 ----
st.markdown('<p class="app-footer">Made with ❤️ · KnowNote 智能笔记助手 · Powered by Streamlit + ECharts + LLM</p>',
            unsafe_allow_html=True)

# ============================================================
# 跨 iframe 通信桥接
# ============================================================
st.components.v1.html("""
<script>
(function() {
    function triggerRerun() {
        var buttons = window.parent.document.querySelectorAll('button');
        for (var i = 0; i < buttons.length; i++) {
            if (buttons[i].textContent.trim() === '⟳') {
                buttons[i].click(); return true;
            }
        }
        return false;
    }
    window.parent.addEventListener('message', function(event) {
        if (!event.data || event.data.type !== 'knownote_click') return;
        var kw = event.data.keyword;
        if (!kw) return;
        var url = new URL(window.parent.location.href);
        url.searchParams.set('keyword', kw);
        window.parent.history.pushState(null, '', url.toString());
        setTimeout(function() {
            if (!triggerRerun()) window.parent.location.href = url.toString();
        }, 50);
    });
    var storedKw = sessionStorage.getItem('knownote_keyword');
    if (storedKw) {
        sessionStorage.removeItem('knownote_keyword');
        var url = new URL(window.parent.location.href);
        if (!url.searchParams.get('keyword')) {
            url.searchParams.set('keyword', storedKw);
            window.parent.history.replaceState(null, '', url.toString());
        }
        setTimeout(function() { triggerRerun(); }, 300);
    }
})();
</script>
""", height=0)
