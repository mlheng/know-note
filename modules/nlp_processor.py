# modules/nlp_processor.py
import jieba
import jieba.analyse
import jieba.posseg as pseg
import re
from typing import List, Tuple, Dict


class NLPProcessor:
    def __init__(self, user_dict_path=None):
        """初始化 NLP 处理器"""
        # 加载自定义词典（可选，用于专业领域）
        if user_dict_path:
            jieba.load_userdict(user_dict_path)

        # 加载停用词表
        self.stopwords = self._load_stopwords()

    def _load_stopwords(self) -> set:
        """加载停用词表"""
        # 常见停用词
        common_stopwords = {'的', '了', '是', '在', '我', '他', '她', '它', '这', '那',
                            '有', '和', '与', '或', '但', '也', '都', '还', '就', '要',
                            '会', '能', '可以', '可能', '虽然', '但是', '所以', '因此'}
        return common_stopwords

    def segment(self, text: str) -> List[str]:
        """中文分词"""
        # 清洗文本
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
        # 精确模式分词
        words = jieba.lcut(text)
        # 过滤停用词和单字
        words = [w for w in words if w not in self.stopwords and len(w.strip()) > 1]
        return words

    def extract_keywords_textrank(self, text: str, topK: int = 10) -> List[Tuple[str, float]]:
        """
        使用 TextRank 算法提取关键词
        TextRank 是基于图排序的无监督算法，效果优于 TF-IDF
        """
        # TextRank 提取，带权重返回
        keywords = jieba.analyse.textrank(
            text,
            topK=topK,
            withWeight=True,
            allowPOS=('n', 'nr', 'ns', 'nt', 'nz', 'v')  # 保留名词类
        )
        return keywords

    def extract_keywords_tfidf(self, text: str, topK: int = 10) -> List[Tuple[str, float]]:
        """使用 TF-IDF 算法提取关键词（备用方案）"""
        keywords = jieba.analyse.extract_tags(
            text,
            topK=topK,
            withWeight=True
        )
        return keywords

    def extract_entities(self, text: str) -> List[Dict]:
        """
        提取命名实体（人名、地名、机构名等）
        使用 jieba 词性标注实现轻量级实体识别
        """
        words = pseg.cut(text)
        entities = []

        # 常见的命名实体词性标记
        entity_types = {
            'nr': '人名',  # 人名
            'ns': '地名',  # 地名
            'nt': '机构名',  # 机构名
            'nz': '专有名词',  # 其他专名
        }

        for word, flag in words:
            if flag in entity_types:
                entities.append({
                    'word': word,
                    'type': entity_types[flag],
                    'pos': flag
                })

        return entities

    def build_cooccurrence_matrix(self, words: List[str], window_size: int = 5) -> Dict[Tuple[str, str], int]:
        """
        构建词共现矩阵
        在滑动窗口内同时出现的词，认为存在关系
        """
        cooccurrence = {}
        for i, word in enumerate(words):
            # 滑动窗口
            for j in range(i + 1, min(i + window_size + 1, len(words))):
                pair = tuple(sorted([word, words[j]]))
                if pair[0] != pair[1]:  # 排除相同词
                    cooccurrence[pair] = cooccurrence.get(pair, 0) + 1
        return cooccurrence