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
if "mindmap_html" not in st.session_state:
    st.session_state.mindmap_html = None
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "🗺️ 知识图谱"


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

        # 3.5 生成思维导图（NLP 模式，始终生成，速度快无成本）
        builder_mm = MindMapBuilder()
        nlp_tree = builder_mm.build_tree_from_graph(
            builder.graph, keywords
        )
        st.session_state.mindmap_html = builder_mm.generate_echarts_tree_html(nlp_tree)

        # 4. 可选：AI 增强（摘要 + 思维导图）
        if enable_llm:
            if not api_key:
                st.warning("请在侧边栏输入 API Key 以启用 AI 增强")
            else:
                with st.spinner("正在生成 AI 摘要和思维导图..."):
                    try:
                        llm = LLMProcessor(api_key, api_type=api_type)

                        # 4a. AI 摘要
                        result = llm.summarize(text)
                        st.session_state.summary = result

                        # 4b. AI 思维导图（LLM 模式覆盖 NLP 版本）
                        mm_result = llm.generate_mindmap(text)
                        if mm_result.get("success"):
                            st.session_state.mindmap_html = builder_mm.generate_markmap_html(
                                mm_result["content"]
                            )
                        # 失败时保留 NLP 版本（已在上面生成）
                    except Exception as e:
                        st.session_state.summary = {
                            "success": False,
                            "content": f"AI 增强异常: {str(e)}"
                        }
                        # mindmap_html 保留 NLP 版本，无需额外处理


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


def export_mindmap_as_html():
    """将当前思维导图导出为独立的 HTML 文件"""
    if "mindmap_html" in st.session_state and st.session_state.mindmap_html:
        import base64
        html_bytes = st.session_state.mindmap_html.encode("utf-8")
        b64 = base64.b64encode(html_bytes).decode()
        href = f'<a href="data:text/html;base64,{b64}" download="mindmap.html">📥 下载思维导图 HTML</a>'
        st.markdown(href, unsafe_allow_html=True)
        st.success("点击上方链接即可下载")
    else:
        st.warning("请先生成思维导图")


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
    enable_llm = st.checkbox("启用 AI 增强（摘要/详解/联想）")
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
        if st.button("导出图谱 HTML", use_container_width=True):
            export_graph_as_html()
    with col_export2:
        if st.button("导出导图 HTML", use_container_width=True):
            export_mindmap_as_html()

    # ---- 隐藏的桥接按钮（JS 点击以触发 Streamlit rerun） ----
    st.divider()
    bridge_clicked = st.button("·", key="__bridge_btn__",
                                help="bridge",
                                use_container_width=True)
    if bridge_clicked:
        st.rerun()

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
    # ---- 视图切换 ----
    col2_header, col2_toggle = st.columns([2, 1])
    with col2_header:
        st.subheader("📊 知识可视化")
    with col2_toggle:
        view_mode = st.radio(
            "视图",
            ["🗺️ 知识图谱", "🧠 思维导图"],
            horizontal=True,
            key="view_mode",
            label_visibility="collapsed"
        )

    st.caption("💡 可在知识图谱和思维导图之间切换，无需重新生成")

    if view_mode == "🗺️ 知识图谱":
        if st.session_state.has_result and st.session_state.graph_html:
            st.components.v1.html(st.session_state.graph_html, height=580)
        else:
            st.info("👈 在左侧输入笔记，点击「生成知识图谱」按钮")
    else:  # 思维导图
        if st.session_state.has_result and st.session_state.mindmap_html:
            st.components.v1.html(st.session_state.mindmap_html, height=580)
        else:
            st.info("👈 在左侧输入笔记，点击「生成知识图谱」按钮")

    # ---- 关键词点击查询结果（独立于 has_result，支持从图谱节点点击跳转后查询） ----
    if clicked_keyword:
        # 去重：同一关键词不重复查询，除非上次查询失败（如缺 API Key）
        last_queried = st.session_state.get("_last_queried_keyword", "")
        prev_info = st.session_state.get("keyword_info")
        prev_failed = prev_info is not None and not prev_info.get("success", True)
        if clicked_keyword != last_queried or prev_info is None or prev_failed:
            st.session_state._last_queried_keyword = clicked_keyword
            with st.spinner(f"正在查询「{clicked_keyword}」的 AI 详解与知识联想..."):
                result = query_keyword_info(clicked_keyword, api_key, api_type)
                st.session_state.keyword_info = result

        # 显示关键词查询结果
        if st.session_state.keyword_info:
            result = st.session_state.keyword_info
            if result.get("success"):
                if not st.session_state.has_result:
                    st.info("💡 图谱数据已过期，但 AI 查询结果已生成。重新输入笔记并生成即可恢复图谱。")
                st.success(f"✅ 「**{clicked_keyword}**」AI 详解与知识联想已生成")
                with st.expander(f"🤖 「{clicked_keyword}」详解与联想", expanded=True):
                    st.markdown(result["content"])
                    # 清除按钮
                    if st.button("❌ 关闭", key="clear_keyword_info"):
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
        st.caption("点击图谱节点或下方按钮可查询 AI 详解与知识联想")
        cols = st.columns(5)
        for i, kw in enumerate(st.session_state.keywords[:15]):
            with cols[i % 5]:
                # 用 link_button 模拟，点击后设置 query_param
                if st.button(kw, key=f"kw_{i}", use_container_width=True,
                             help=f"点击查询「{kw}」的 AI 详解与知识联想"):
                    st.query_params["keyword"] = kw
                    st.rerun()

    # ---- 图谱统计 ----
    if st.session_state.has_result:
        with st.expander("📊 图谱统计"):
            kw_count = len(st.session_state.keywords)
            st.metric("关键词节点数", kw_count)
            st.caption("关系强度阈值: " + str(edge_threshold))

# ============================================================
# 跨 iframe 通信桥接（放在页面底部，确保所有元素已渲染）
# 原理：图谱/导图 iframe → postMessage → 桥接 iframe 监听 parent window
#       → pushState 更新 URL（无重载）→ 点击隐藏的 Streamlit 按钮 → st.rerun()
# ============================================================
st.components.v1.html("""
<script>
(function() {
    function triggerStreamlitRerun() {
        // 在父页面中查找 bridge 按钮并点击
        var buttons = window.parent.document.querySelectorAll('button');
        for (var i = 0; i < buttons.length; i++) {
            if (buttons[i].textContent.indexOf('__bridge__') >= 0 ||
                buttons[i].getAttribute('aria-label') === 'bridge') {
                buttons[i].click();
                console.log('KNOWNOTE-BRIDGE: clicked bridge button');
                return true;
            }
        }
        console.log('KNOWNOTE-BRIDGE: bridge button not found');
        return false;
    }

    // 在 parent window 上注册监听器
    // 图谱/导图 iframe 发送 parent.postMessage → parent 接收 → 此回调触发
    window.parent.addEventListener('message', function(event) {
        if (!event.data || event.data.type !== 'knownote_click') return;
        var keyword = event.data.keyword;
        if (!keyword) return;
        console.log('KNOWNOTE-BRIDGE: received keyword:', keyword);

        // 1) pushState 更新 URL（不触发页面重载）
        var url = new URL(window.parent.location.href);
        url.searchParams.set('keyword', keyword);
        window.parent.history.pushState(null, '', url.toString());

        // 2) 尝试点击 Streamlit bridge 按钮触发 rerun
        setTimeout(function() {
            if (!triggerStreamlitRerun()) {
                // 回退：全页跳转
                console.log('KNOWNOTE-BRIDGE: fallback to full reload');
                window.parent.location.href = url.toString();
            }
        }, 50);
    });

    // 页面加载时，检查 sessionStorage 中是否有待处理的关键词
    var storedKw = sessionStorage.getItem('knownote_keyword');
    if (storedKw) {
        sessionStorage.removeItem('knownote_keyword');
        var url = new URL(window.parent.location.href);
        if (!url.searchParams.get('keyword')) {
            url.searchParams.set('keyword', storedKw);
            window.parent.history.replaceState(null, '', url.toString());
        }
        setTimeout(function() { triggerStreamlitRerun(); }, 300);
    }
})();
</script>
""", height=0)
