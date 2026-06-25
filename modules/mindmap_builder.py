# modules/mindmap_builder.py
import json
from collections import deque
from typing import List, Tuple, Dict, Optional


class MindMapBuilder:
    """
    思维导图构建器，支持两种模式：
    1. NLP 模式：从 NetworkX 共现图谱构建层级树
    2. 通用模式：将 markdown 字符串解析为树结构

    可视化基于 ECharts tree 组件（与知识图谱使用同一套 CDN，稳定可靠）
    """

    # ------------------------------------------------------------------
    # 树结构构建
    # ------------------------------------------------------------------

    def build_tree_from_graph(self,
                               graph,
                               keywords: List[Tuple[str, float]],
                               max_depth: int = 4,
                               max_children: int = 6) -> dict:
        """
        从 NetworkX 共现图谱构建层级树（NLP 模式，免费）

        Args:
            graph: networkx.Graph，节点属性含 'name'/'weight'，边属性含 'weight'
            keywords: TextRank/TF-IDF 关键词列表 [(word, score), ...]，已按 score 降序
            max_depth: 最大层级深度
            max_children: 每个节点最多保留的子节点数

        Returns:
            ECharts tree 格式的 dict: {"name": "...", "children": [...]}
        """
        if graph is None or graph.number_of_nodes() == 0:
            return {"name": "(无内容)"}

        # ---- Step 1: 选择根节点 ----
        root_id = None

        # 取 TextRank 评分最高的关键词作为根
        for kw_name, _ in keywords:
            for nid in graph.nodes:
                if graph.nodes[nid].get('name') == kw_name:
                    root_id = nid
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

        # 最终回退
        if root_id is None:
            root_id = list(graph.nodes)[0]

        # ---- Step 2: BFS 构建层级树 ----
        visited = {root_id}
        tree_adj: Dict[str, list] = {root_id: []}
        queue = deque([(root_id, 0)])

        while queue:
            current_id, depth = queue.popleft()
            if depth >= max_depth:
                continue

            candidates = []
            for neighbor_id in graph.neighbors(current_id):
                if neighbor_id not in visited:
                    w = graph.edges[current_id, neighbor_id].get('weight', 0)
                    candidates.append((neighbor_id, w))

            candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = candidates[:max_children]

            for neighbor_id, _ in candidates:
                visited.add(neighbor_id)
                tree_adj.setdefault(current_id, []).append(neighbor_id)
                tree_adj.setdefault(neighbor_id, [])
                queue.append((neighbor_id, depth + 1))

        # ---- Step 3: DFS 构建 ECharts tree JSON ----
        def _node_name(nid):
            return graph.nodes[nid].get('name', str(nid))

        def _build_node(nid):
            node = {"name": _node_name(nid)}
            children = tree_adj.get(nid, [])
            if children:
                node["children"] = [_build_node(c) for c in children]
            return node

        return _build_node(root_id)

    def parse_markdown_to_tree(self, markdown_content: str) -> dict:
        """
        将 markdown 标题大纲解析为 ECharts tree 数据格式

        Args:
            markdown_content: 包含 # ## ### #### 标题的 markdown 字符串

        Returns:
            ECharts tree 格式的 dict
        """
        lines = markdown_content.strip().split('\n')
        if not lines:
            return {"name": "(无内容)"}

        # 解析每一行，提取层级和文本
        parsed_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # 计算 # 开头的层级
            level = 0
            for ch in stripped:
                if ch == '#':
                    level += 1
                else:
                    break
            if level > 0:
                text = stripped[level:].strip()
                if text:
                    parsed_lines.append((level, text))

        if not parsed_lines:
            # 没有标题行，将所有内容作为单个根节点
            return {"name": markdown_content.strip()[:50]}

        # 使用栈构建树
        # 确保第一个标题是 #（level 1）
        min_level = min(lvl for lvl, _ in parsed_lines)
        root = None
        stack: List[Tuple[int, dict]] = []  # (level, node)

        for level, text in parsed_lines:
            # 归一化层级（使最小层级变为 1）
            normalized_level = level - min_level + 1
            if normalized_level > 4:
                normalized_level = 4

            node = {"name": text}

            if not stack:
                # 第一个节点作为根
                root = node
                stack.append((normalized_level, node))
                continue

            # 找到正确的父节点
            while stack and stack[-1][0] >= normalized_level:
                stack.pop()

            if stack:
                parent = stack[-1][1]
                parent.setdefault("children", []).append(node)
            else:
                # 栈空了，说明层级异常，作为根的兄弟
                root = node

            stack.append((normalized_level, node))

        return root if root else {"name": "(无内容)"}

    # ------------------------------------------------------------------
    # ECharts Tree HTML 生成
    # ------------------------------------------------------------------

    def generate_echarts_tree_html(self,
                                    tree_data: dict,
                                    height: int = 550) -> str:
        """
        将树数据渲染为 ECharts tree 思维导图 HTML

        Args:
            tree_data: {"name": "...", "children": [...]} 格式的树数据
            height: iframe 高度

        Returns:
            完整的 HTML 文档字符串
        """
        tree_json = json.dumps(tree_data, ensure_ascii=False)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ background: #f8f9fa; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif; }}
        #mindmap {{ width: 100%; height: {height}px; }}
        #click-hint {{
            text-align: center;
            color: #888;
            font-size: 12px;
            margin-top: 4px;
            display: none;
        }}
    </style>
</head>
<body>
    <div id="mindmap"></div>
    <div id="click-hint">💡 点击任意节点 → AI 详解与知识联想</div>
    <script>
        (function() {{
            var chartDom = document.getElementById('mindmap');
            var myChart = echarts.init(chartDom);

            var treeData = {tree_json};

            var option = {{
                tooltip: {{
                    trigger: 'item',
                    formatter: function(params) {{
                        var hasChildren = params.data && params.data.children && params.data.children.length > 0;
                        var hint = hasChildren
                            ? '<br/><span style="color:#888;font-size:11px;">🖱️ 点击查询 AI 详解与知识联想</span>'
                            : '<br/><span style="color:#888;font-size:11px;">🖱️ 点击查询 AI 详解与知识联想 →</span>';
                        return '<b>' + params.name + '</b>' + hint;
                    }}
                }},
                series: [{{
                    type: 'tree',
                    data: [treeData],
                    layout: 'orthogonal',
                    orient: 'LR',
                    roam: true,
                    expandAndCollapse: false,
                    initialTreeDepth: -1,
                    symbol: 'roundRect',
                    symbolSize: [18, 18],
                    edgeShape: 'curve',
                    edgeForkPosition: '50%',
                    itemStyle: {{
                        borderColor: '#1890ff',
                        borderWidth: 2,
                        color: '#e6f7ff'
                    }},
                    lineStyle: {{
                        color: '#91d5ff',
                        width: 2,
                        curveness: 0.5
                    }},
                    label: {{
                        position: 'right',
                        verticalAlign: 'middle',
                        align: 'left',
                        fontSize: 13,
                        color: '#333',
                        fontWeight: 'bold'
                    }},
                    emphasis: {{
                        itemStyle: {{
                            borderColor: '#096dd9',
                            borderWidth: 3,
                            shadowBlur: 15,
                            shadowColor: 'rgba(24,144,255,0.4)'
                        }},
                        label: {{
                            fontSize: 15,
                            fontWeight: 'bold'
                        }},
                        lineStyle: {{
                            width: 3,
                            color: '#1890ff'
                        }}
                    }},
                    leaves: {{
                        label: {{
                            position: 'right',
                            verticalAlign: 'middle',
                            align: 'left'
                        }},
                        itemStyle: {{
                            borderColor: '#52c41a',
                            color: '#f6ffed'
                        }}
                    }}
                }}]
            }};

            myChart.setOption(option);

            // ========================================
            // 单击节点 → AI 查询（通过 URL query 传递关键词给 Streamlit）
            // 注：ECharts tree 的 params.dataType 为 undefined，
            //     直接用 params.name 判断是否为节点点击
            // ========================================
            myChart.on('click', function(params) {{
                if (params.name) {{
                    var keyword = params.name;
                    try {{
                        var currentUrl = window.parent.location.href;
                        var baseUrl = currentUrl.split('?')[0];
                        var newUrl = baseUrl + '?keyword=' + encodeURIComponent(keyword);
                        window.parent.location.href = newUrl;
                    }} catch(e) {{
                        // fallback: 修改当前窗口 URL（srcdoc iframe 同源场景不会到这里）
                        var currentUrl = window.location.href;
                        var baseUrl = currentUrl.split('?')[0];
                        var newUrl = baseUrl + '?keyword=' + encodeURIComponent(keyword);
                        window.location.href = newUrl;
                    }}
                }}
            }});

            // 鼠标悬停时改变光标样式（tree 系列无需 dataType 判断，都是节点）
            myChart.on('mouseover', function(params) {{
                if (params.name) {{
                    chartDom.style.cursor = 'pointer';
                }}
            }});
            myChart.on('mouseout', function(params) {{
                chartDom.style.cursor = 'default';
            }});

            // 自适应大小
            window.addEventListener('resize', function() {{
                myChart.resize();
            }});

            // 显示操作提示
            setTimeout(function() {{
                var hint = document.getElementById('click-hint');
                if (hint) hint.style.display = 'block';
            }}, 1500);
        }})();
    </script>
</body>
</html>
"""
        return html

    # ------------------------------------------------------------------
    # 便捷方法（保持向后兼容）
    # ------------------------------------------------------------------

    def build_markdown_from_graph(self,
                                   graph,
                                   keywords: List[Tuple[str, float]],
                                   max_depth: int = 4,
                                   max_children: int = 6) -> str:
        """
        从 NetworkX 共现图谱构建 markdown 层级大纲

        保留此方法以供 LLM 模式下调试使用
        """
        if graph is None or graph.number_of_nodes() == 0:
            return "# (无内容)\n"

        root_id = None
        for kw_name, _ in keywords:
            for nid in graph.nodes:
                if graph.nodes[nid].get('name') == kw_name:
                    root_id = nid
                    break
            if root_id is not None:
                break

        if root_id is None:
            max_deg = -1
            for nid in graph.nodes:
                deg = graph.degree(nid)
                if deg > max_deg:
                    max_deg = deg
                    root_id = nid

        if root_id is None:
            root_id = list(graph.nodes)[0]

        visited = {root_id}
        tree_adj: Dict[str, list] = {root_id: []}
        queue = deque([(root_id, 0)])

        while queue:
            current_id, depth = queue.popleft()
            if depth >= max_depth:
                continue

            candidates = []
            for neighbor_id in graph.neighbors(current_id):
                if neighbor_id not in visited:
                    w = graph.edges[current_id, neighbor_id].get('weight', 0)
                    candidates.append((neighbor_id, w))

            candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = candidates[:max_children]

            for neighbor_id, _ in candidates:
                visited.add(neighbor_id)
                tree_adj.setdefault(current_id, []).append(neighbor_id)
                tree_adj.setdefault(neighbor_id, [])
                queue.append((neighbor_id, depth + 1))

        def _node_name(nid):
            return graph.nodes[nid].get('name', str(nid))

        def _dfs_to_markdown(nid, level=1):
            if level > 4:
                return ''
            lines = [f"{'#' * level} {_node_name(nid)}"]
            for child_id in tree_adj.get(nid, []):
                child_md = _dfs_to_markdown(child_id, level + 1)
                if child_md:
                    lines.append(child_md)
            return '\n'.join(lines)

        return _dfs_to_markdown(root_id)

    def generate_markmap_html(self, markdown_content: str, height: int = 550) -> str:
        """
        将 markdown 内容渲染为 ECharts tree 思维导图 HTML

        （方法名保留 generate_markmap_html 以保证向后兼容，
          实际已改用 ECharts tree 渲染）
        """
        tree_data = self.parse_markdown_to_tree(markdown_content)
        return self.generate_echarts_tree_html(tree_data, height=height)
