# modules/mindmap_builder.py
import json
from collections import deque
from typing import List, Tuple, Dict


class MindMapBuilder:
    """
    思维导图构建器，支持两种模式：
    1. NLP 模式：从 NetworkX 共现图谱构建层级树，生成 markdown 大纲
    2. 通用模式：将任意 markdown 字符串渲染为 markmap 交互式思维导图 HTML
    """

    def build_markdown_from_graph(self,
                                   graph,
                                   keywords: List[Tuple[str, float]],
                                   max_depth: int = 4,
                                   max_children: int = 6) -> str:
        """
        从 NetworkX 共现图谱构建 markdown 层级大纲（NLP 模式，免费）

        Args:
            graph: networkx.Graph，节点属性含 'name'/'weight'，边属性含 'weight'
            keywords: TextRank/TF-IDF 关键词列表 [(word, score), ...]，已按 score 降序
            max_depth: 最大层级深度（对应 # ~ ####）
            max_children: 每个节点最多保留的子节点数

        Returns:
            markdown 字符串，以 # ## ### #### 表示层级
        """
        if graph is None or graph.number_of_nodes() == 0:
            return "# (无内容)\n"

        # ---- Step 1: 选择根节点 ----
        root_id = None
        root_name = None

        # 取 TextRank 评分最高的关键词作为根
        for kw_name, _ in keywords:
            for nid in graph.nodes:
                if graph.nodes[nid].get('name') == kw_name:
                    root_id = nid
                    root_name = kw_name
                    break
            if root_id is not None:
                break

        # 回退：取度最大的节点
        if root_id is None:
            max_deg = -1
            for nid in graph.nodes:
                deg = graph.degree(nid)
                if deg > max_deg:
                    max_deg = deg
                    root_id = nid
                    root_name = graph.nodes[nid].get('name', str(nid))

        # 最终回退
        if root_id is None:
            root_id = list(graph.nodes)[0]
            root_name = graph.nodes[root_id].get('name', str(root_id))

        # ---- Step 2: BFS 构建层级树 ----
        visited = {root_id}
        tree_adj: Dict[str, list] = {root_id: []}
        queue = deque([(root_id, 0)])

        while queue:
            current_id, depth = queue.popleft()
            if depth >= max_depth:
                continue

            # 收集未访问的邻居及其边权重
            candidates = []
            for neighbor_id in graph.neighbors(current_id):
                if neighbor_id not in visited:
                    w = graph.edges[current_id, neighbor_id].get('weight', 0)
                    candidates.append((neighbor_id, w))

            # 按边权重降序排列，限制子节点数量
            candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = candidates[:max_children]

            for neighbor_id, _ in candidates:
                visited.add(neighbor_id)
                tree_adj.setdefault(current_id, []).append(neighbor_id)
                tree_adj.setdefault(neighbor_id, [])
                queue.append((neighbor_id, depth + 1))

        # ---- Step 3: DFS 生成 markdown ----
        def _node_name(nid):
            return graph.nodes[nid].get('name', str(nid))

        def _dfs_to_markdown(nid, level=1):
            if level > 4:
                return ''
            lines = [f"{'#' * level} {_node_name(nid)}"]
            children = tree_adj.get(nid, [])
            for child_id in children:
                child_md = _dfs_to_markdown(child_id, level + 1)
                if child_md:
                    lines.append(child_md)
            return '\n'.join(lines)

        markdown = _dfs_to_markdown(root_id)
        return markdown

    def generate_markmap_html(self, markdown_content: str, height: int = 550) -> str:
        """
        将 markdown 内容渲染为 markmap 交互式思维导图 HTML

        Args:
            markdown_content: markdown 字符串（用 # ## ### 表示层级）
            height: iframe 高度（像素）

        Returns:
            完整的 HTML 文档字符串，可直接用于 st.components.v1.html()
        """
        # 用 json.dumps 安全嵌入 markdown（处理引号、换行、反斜杠等）
        md_json = json.dumps(markdown_content, ensure_ascii=False)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ background: #f8f9fa; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif; }}
        #mindmap {{ width: 100%; height: {height}px; }}
        .markmap-toolbar {{ opacity: 0.7; transition: opacity 0.2s; }}
        .markmap-toolbar:hover {{ opacity: 1; }}
    </style>
</head>
<body>
    <div id="mindmap"></div>

    <!-- D3 依赖（markmap 底层基于 D3） -->
    <script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js">
    </script>
    <!-- markmap 渲染引擎 -->
    <script src="https://cdn.jsdelivr.net/npm/markmap-view@0.17.0/dist/browser/index.min.js">
    </script>
    <!-- markmap 工具栏（缩放/适配/重置） -->
    <script src="https://cdn.jsdelivr.net/npm/markmap-toolbar@0.17.0/dist/browser/index.min.js">
    </script>

    <script>
    (function() {{
        // 嵌入的 markdown 内容（通过 JSON 编码安全注入）
        var markdown = {md_json};

        // markmap 暴露的全局对象
        var Transformer = markmap.Transformer;
        var Markmap = markmap.Markmap;
        var Toolbar = markmap.Toolbar;

        // 将 markdown 转换为 markmap 内部数据
        var transformer = new Transformer();
        var result = transformer.transform(markdown);

        // 渲染思维导图
        var mm = Markmap.create(
            document.getElementById('mindmap'),
            {{
                autoFit: true,
                colorFreezeLevel: 2,
                duration: 400,
                maxWidth: 280,
                paddingX: 16,
                spacingHorizontal: 80,
                spacingVertical: 10,
                initialExpandLevel: 2
            }},
            result.root
        );

        // 附加工具栏
        Toolbar.create(mm);

        // 窗口大小变化时自适应
        window.addEventListener('resize', function() {{
            mm.fit();
        }});
    }})();
    </script>
</body>
</html>
"""
        return html
