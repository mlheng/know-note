# modules/graph_builder.py
import networkx as nx
from typing import List, Dict, Tuple, Any
import json


class KnowledgeGraphBuilder:
    def __init__(self):
        """初始化知识图谱构建器"""
        self.graph = nx.Graph()
        self.node_counter = 0
        self.node_ids = {}  # 节点名称到ID的映射

    def _get_node_id(self, node_name: str) -> str:
        """获取或创建节点ID"""
        if node_name not in self.node_ids:
            self.node_counter += 1
            self.node_ids[node_name] = f"node_{self.node_counter}"
        return self.node_ids[node_name]

    def build_from_cooccurrence(self,
                                keywords: List[Tuple[str, float]],
                                cooccurrence: Dict[Tuple[str, str], int],
                                edge_threshold: float = 0.3) -> None:
        """
        基于关键词共现构建图谱
        keywords: 关键词及权重列表
        cooccurrence: 共现矩阵
        edge_threshold: 边权重阈值（归一化后）
        """
        self.graph.clear()
        self.node_ids.clear()
        self.node_counter = 0

        # 添加节点（关键词）
        for word, weight in keywords:
            node_id = self._get_node_id(word)
            self.graph.add_node(node_id, name=word, weight=weight, size=weight * 50)

        # 计算最大共现次数，用于归一化
        if not cooccurrence:
            return
        max_count = max(cooccurrence.values())

        # 添加边
        for (word1, word2), count in cooccurrence.items():
            # 只有两个词都在关键词列表中才添加边
            if word1 in self.node_ids and word2 in self.node_ids:
                normalized_weight = count / max_count if max_count > 0 else 0
                if normalized_weight >= edge_threshold:
                    self.graph.add_edge(
                        self.node_ids[word1],
                        self.node_ids[word2],
                        weight=normalized_weight,
                        count=count
                    )

    def build_from_entities(self,
                            entities: List[Dict],
                            text: str = None) -> None:
        """
        基于命名实体构建图谱
        """
        self.graph.clear()
        self.node_ids.clear()
        self.node_counter = 0

        # 添加实体节点
        for entity in entities:
            node_id = self._get_node_id(entity['word'])
            self.graph.add_node(
                node_id,
                name=entity['word'],
                entity_type=entity['type'],
                size=30
            )

        # 简单策略：按位置顺序连接相邻实体
        for i, entity_i in enumerate(entities):
            for j, entity_j in enumerate(entities[i + 1:]):
                position_weight = 1.0 / (abs(i - (i + j + 1)) + 1)
                if entity_i['word'] in self.node_ids and entity_j['word'] in self.node_ids:
                    self.graph.add_edge(
                        self.node_ids[entity_i['word']],
                        self.node_ids[entity_j['word']],
                        weight=position_weight
                    )

    def to_echarts_format(self) -> Dict:
        """
        将 NetworkX 图转换为 ECharts 可识别的格式
        """
        # 按实体类型分类
        categories_map = {}
        categories = []
        node_categories = {}

        for node_id in self.graph.nodes:
            node_data = self.graph.nodes[node_id]
            node_name = node_data.get('name', node_id)

            if 'entity_type' in node_data:
                cat_name = node_data['entity_type']
            else:
                cat_name = "概念"

            if cat_name not in categories_map:
                categories_map[cat_name] = len(categories)
                categories.append({"name": cat_name})

            node_categories[node_name] = categories_map[cat_name]

        # 构建节点列表
        nodes = []
        for node_id in self.graph.nodes:
            node_data = self.graph.nodes[node_id]
            node_name = node_data.get('name', node_id)
            size = node_data.get('size', node_data.get('weight', 0.5) * 50 + 20)

            nodes.append({
                "name": node_name,
                "category": node_categories.get(node_name, 0),
                "symbolSize": max(20, min(size, 80)),
                "value": node_data.get('weight', 0.5),
                "label": {"show": True}
            })

        # 构建边列表
        links = []
        for u, v, edge_data in self.graph.edges(data=True):
            u_name = self.graph.nodes[u].get('name', u)
            v_name = self.graph.nodes[v].get('name', v)
            links.append({
                "source": u_name,
                "target": v_name,
                "value": edge_data.get('weight', 0.5)
            })

        return {
            "categories": categories,
            "nodes": nodes,
            "links": links
        }

    def generate_echarts_html(self) -> str:
        """
        生成包含 ECharts 关系图的完整 HTML 代码
        节点可点击，点击后通过 URL query 参数传递给 Streamlit 触发 LLM 查询
        """
        chart_data = self.to_echarts_format()

        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
            <style>
                #main {{ width: 100%; height: 550px; cursor: default; }}
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
            <div id="main"></div>
            <div id="click-hint">💡 点击任意节点 → AI 详解 + 知识联想</div>
            <script>
                var chartDom = document.getElementById('main');
                var myChart = echarts.init(chartDom);

                var option = {{
                    title: {{
                        text: '📚 知识图谱',
                        subtext: '点击节点 → AI 详解与知识联想 | 可拖拽/缩放',
                        left: 'center',
                        top: 5,
                        subtextStyle: {{ fontSize: 11, color: '#999' }}
                    }},
                    tooltip: {{
                        trigger: 'item',
                        formatter: function(params) {{
                            if (params.dataType === 'node') {{
                                return '<b>' + params.name + '</b><br/><span style="color:#888;">🖱️ 点击查询 AI 详解与知识联想 →</span>';
                            }}
                            return params.data.source + ' → ' + params.data.target;
                        }}
                    }},
                    series: [{{
                        type: 'graph',
                        layout: 'force',
                        roam: true,
                        draggable: true,
                        data: {json.dumps(chart_data['nodes'], ensure_ascii=False)},
                        links: {json.dumps(chart_data['links'], ensure_ascii=False)},
                        categories: {json.dumps(chart_data['categories'], ensure_ascii=False)},
                        label: {{
                            show: true,
                            position: 'right',
                            fontSize: 12
                        }},
                        emphasis: {{
                            scale: true,
                            focus: 'adjacency',
                            itemStyle: {{
                                borderColor: '#1890ff',
                                borderWidth: 3,
                                shadowBlur: 20,
                                shadowColor: 'rgba(24,144,255,0.5)'
                            }},
                            label: {{
                                show: true,
                                fontSize: 15,
                                fontWeight: 'bold'
                            }}
                        }},
                        force: {{
                            repulsion: 500,
                            edgeLength: [50, 100],
                            gravity: 0.1,
                            friction: 0.6,
                            layoutAnimation: true
                        }},
                        lineStyle: {{
                            color: 'source',
                            curveness: 0.3,
                            opacity: 0.7
                        }},
                        roamZoom: true,
                        roamPan: true
                    }}]
                }};

                myChart.setOption(option);

                // ========================================
                // 节点点击事件 — 通过 postMessage 桥接到 Streamlit（无页面重载）
                // ========================================
                myChart.on('click', function(params) {{
                    if (params.dataType === 'node') {{
                        var keyword = params.name;
                        var hint = document.getElementById('click-hint');
                        console.log('KNOWNOTE-KG: node clicked:', keyword);

                        // 视觉反馈
                        if (hint) {{
                            hint.textContent = '✅ 正在查询: ' + keyword + ' ...';
                            hint.style.color = '#1890ff';
                            hint.style.fontWeight = 'bold';
                            hint.style.display = 'block';
                        }}

                        // 通过 postMessage 发送关键词给桥接 iframe
                        // 桥接 iframe 负责更新 URL(pushState) + 触发 Streamlit rerun
                        window.parent.postMessage({{
                            type: 'knownote_click',
                            keyword: keyword
                        }}, '*');

                        // 兜底：1.5s 后如果 postMessage 没有触发 rerun，用传统跳转
                        var fallbackTimer = setTimeout(function() {{
                            if (hint && hint.style.color === 'rgb(24, 144, 255)') {{
                                hint.textContent = '⚠️ 正在尝试备用跳转方式...';
                                hint.style.color = '#faad14';
                                var newUrl = window.parent.location.href.split('?')[0]
                                           + '?keyword=' + encodeURIComponent(keyword);
                                window.parent.location.href = newUrl;
                            }}
                        }}, 1500);

                        // 如果 postMessage 成功触发 rerun，清除兜底定时器
                        // （rerun 会导致页面重绘，定时器自然失效）
                    }}
                }});

                // 鼠标悬停时改变光标样式
                myChart.on('mouseover', function(params) {{
                    if (params.dataType === 'node') {{
                        chartDom.style.cursor = 'pointer';
                    }}
                }});
                myChart.on('mouseout', function(params) {{
                    if (params.dataType === 'node') {{
                        chartDom.style.cursor = 'default';
                    }}
                }});

                // 自适应大小
                window.addEventListener('resize', function() {{
                    myChart.resize();
                }});

                // 显示点击提示
                setTimeout(function() {{
                    var hint = document.getElementById('click-hint');
                    if (hint) hint.style.display = 'block';
                }}, 1500);
            </script>
        </body>
        </html>
        """

        return html_template
