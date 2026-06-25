# modules/llm_processor.py
import requests
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class LLMProcessor:
    """
    大模型处理器，支持多种 API 后端
    - openai: OpenAI API (或兼容的代理)
    - deepseek: DeepSeek API (价格极低)
    - baidu: 百度 ERNIE-Speed (有免费额度)
    """

    # 各平台的 API 配置
    API_CONFIG = {
        "openai": {
            "url": "https://api.openai.com/v1/chat/completions",
            "model": "gpt-3.5-turbo",
        },
        "deepseek": {
            "url": "https://api.deepseek.com/chat/completions",
            "model": "deepseek-chat",
        },
        "baidu": {
            "url": "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-speed-128k",
            "model": "ernie-speed-128k",
        },
    }

    def __init__(self, api_key: str = None, api_type: str = "deepseek"):
        """
        初始化大模型处理器

        Args:
            api_key: API 密钥
            api_type: API 类型，支持 'openai' / 'deepseek' / 'baidu'
        """
        self.api_key = api_key
        self.api_type = api_type

        if api_type not in self.API_CONFIG:
            raise ValueError(
                f"不支持的 API 类型: {api_type}，可选值: {list(self.API_CONFIG.keys())}"
            )

    def summarize(self, text: str, max_length: int = 300) -> dict:
        """生成笔记摘要，返回 {success, content} 字典"""
        prompt = f"""请为以下笔记内容生成一个简短的摘要（不超过{max_length}字）：

{text[:2000]}

摘要："""
        return self._call_llm(prompt, max_tokens=500)

    def ask_question(self, text: str, question: str) -> dict:
        """基于笔记内容回答问题，返回 {success, content} 字典"""
        prompt = f"""基于以下笔记内容回答问题：

笔记内容：
{text[:3000]}

问题：{question}

回答："""
        return self._call_llm(prompt, max_tokens=500)

    def get_keyword_info(self, keyword: str, context: str = "") -> dict:
        """
        查询某个关键词的 AI 详解与知识联想，返回 {success, content} 字典

        Args:
            keyword: 要查询的关键词
            context: 可选，用户原始笔记内容作为上下文
        """
        context_part = ""
        if context:
            context_part = f"\n\n用户笔记上下文：\n{context[:1500]}"

        prompt = f"""请对以下关键词进行详细解释，并联想与其相关的知识点：

关键词：{keyword}{context_part}

请从以下几个方面回答（使用 Markdown 格式）：
1. **📖 概念解释**：用通俗易懂的语言解释该关键词的含义（2-3句话）
2. **🔑 核心要点**：列出 3-5 个最重要的知识点或特征
3. **🔗 知识联想**：关联 3-5 个与之密切相关的概念或知识点，并简要说明它们之间的关系
4. **💡 应用场景**：该知识在实际中有哪些典型应用（1-2个例子）
5. **📚 学习路径**：推荐进一步深入学习的方向或资源

要求：内容充实有深度，总字数在 400 字以内，格式清晰易读。"""
        return self._call_llm(prompt, max_tokens=800)

    def extract_keywords_with_llm(self, text: str, top_k: int = 10) -> dict:
        """使用 LLM 提取关键词（比 TextRank 更智能），返回 {success, content} 字典"""
        prompt = f"""请从以下文本中提取 {top_k} 个最重要的关键词，用逗号分隔：

{text[:2000]}

关键词："""
        return self._call_llm(prompt, max_tokens=200)

    def generate_mindmap(self, text: str, max_depth: int = 4) -> dict:
        """
        使用 LLM 生成思维导图结构的 Markdown 大纲

        Args:
            text: 用户笔记内容
            max_depth: 思维导图最大层级深度

        Returns:
            dict: {"success": bool, "content": str}，content 为 markdown 字符串
        """
        prompt = f"""请为以下笔记内容生成一个思维导图结构，以 Markdown 标题格式输出。

要求：
1. 用 `#` 表示中心主题（1个）
2. 用 `##` 表示主要分支（3-6个）
3. 用 `###` 表示次要分支（每个主要分支下2-4个）
4. 最多使用到 `{'#' * max_depth}`（{max_depth}级标题）
5. 每个节点内容不超过15个字，简洁有力
6. 层级关系要体现知识的内在逻辑结构
7. 直接输出 Markdown，不要添加任何解释、前言或后缀

笔记内容：
{text[:2000]}

思维导图 Markdown："""
        return self._call_llm(prompt, max_tokens=800)

    def _call_llm(self, prompt: str, max_tokens: int = 500) -> dict:
        """
        调用大模型 API，根据 api_type 自动路由到对应平台

        Returns:
            dict: {"success": bool, "content": str}
        """
        if not self.api_key:
            return {"success": False, "content": "⚠️ 请先在侧边栏输入 API Key"}

        # 复制 config，避免修改类级别字典（修复 URL 拼接 bug）
        config = dict(self.API_CONFIG[self.api_type])
        headers = self._build_headers()

        # 构建请求 URL（百度需要在 URL 中拼接 access_token）
        url = config["url"]
        if self.api_type == "baidu":
            url = f"{url}?access_token={self.api_key}"

        payload = self._build_payload(config, prompt, max_tokens)

        try:
            logger.info(f"Calling {self.api_type} API: {url[:80]}...")
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=60,
            )

            # 先尝试解析响应体，获取详细错误信息
            try:
                resp_json = response.json()
            except Exception:
                resp_json = {"error": response.text[:300]}

            # 检查 HTTP 状态码
            if not response.ok:
                error_msg = self._format_error(response.status_code, resp_json)
                logger.error(f"API error [{response.status_code}]: {error_msg}")
                return {"success": False, "content": error_msg}

            content = self._parse_response(resp_json)
            return {"success": True, "content": content}

        except requests.exceptions.Timeout:
            msg = f"⏱️ API 请求超时（{self.api_type}），请稍后重试"
            logger.error(msg)
            return {"success": False, "content": msg}
        except requests.exceptions.ConnectionError as e:
            msg = f"🔌 无法连接到 {self.api_type} API\n\n> 请检查：\n> 1. 网络是否正常\n> 2. 是否需要代理\n> 3. API 地址是否正确\n\n> 详情: {str(e)[:200]}"
            logger.error(msg)
            return {"success": False, "content": msg}
        except Exception as e:
            msg = f"❌ API 调用异常: {type(e).__name__}: {str(e)}"
            logger.error(msg)
            return {"success": False, "content": msg}

    def _format_error(self, status_code: int, resp_json: dict) -> str:
        """格式化 API 错误信息"""
        detail = ""

        # 尝试多种常见的错误消息格式
        if isinstance(resp_json, dict):
            error = resp_json.get("error", {})
            if isinstance(error, dict):
                detail = error.get("message", "") or str(error)
            elif isinstance(error, str):
                detail = error
            if not detail:
                detail = resp_json.get("message", "") or resp_json.get("detail", "")

        if not detail:
            detail = str(resp_json)[:300]

        # 常见错误码的友好提示
        tips = {
            401: "\n\n> 💡 API Key 无效或已过期，请检查后重新输入",
            403: "\n\n> 💡 访问被拒绝，请检查 API Key 权限或账户余额",
            429: "\n\n> 💡 请求太频繁或余额不足，请稍后再试",
            500: "\n\n> 💡 API 服务器内部错误，请稍后重试",
            503: "\n\n> 💡 API 服务暂时不可用，请稍后重试",
        }

        tip = tips.get(status_code, "")
        return f"❌ API 请求失败 [{status_code}]: {detail}{tip}"

    def _build_headers(self) -> dict:
        """构建请求头"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, config: dict, prompt: str, max_tokens: int) -> dict:
        """构建请求体"""
        return {
            "model": config["model"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }

    def _parse_response(self, result: dict) -> str:
        """解析各平台的响应格式"""
        # OpenAI 兼容格式（openai / deepseek 都遵循此格式）
        if "choices" in result and len(result["choices"]) > 0:
            choice = result["choices"][0]
            if "message" in choice:
                return choice["message"]["content"]
            if "text" in choice:
                return choice["text"]

        # 百度 ERNIE 格式
        if "result" in result:
            return result["result"]

        return f"⚠️ 无法解析 API 返回: {str(result)[:200]}"
