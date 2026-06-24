# modules/ocr_processor.py
import tempfile
import os
import uuid
import logging

logger = logging.getLogger(__name__)


class OCRProcessor:
    """
    OCR 文字识别处理器，基于 PaddleOCR。
    兼容 PaddleOCR 2.x 和 3.x 两个大版本。
    不依赖 Streamlit，可独立使用。
    """

    def __init__(self):
        self.ocr = None
        self._temp_dir = tempfile.mkdtemp(prefix="knownote_ocr_")
        self._paddleocr_version = 3  # 默认假设 3.x

    def init_engine(self):
        """懒加载 OCR 引擎，失败返回 None"""
        if self.ocr is not None:
            return self.ocr

        try:
            from paddleocr import PaddleOCR

            # PaddleOCR 2.x 和 3.x 的 `lang` 参数都有效，无法用参数区分版本。
            # 用 2.x 参数初始化（2.x 最稳定），失败再回退 3.x。
            try:
                self.ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang="ch",
                    show_log=False,
                )
            except (ValueError, TypeError):
                # 2.x 参数失败，尝试 3.x
                self.ocr = PaddleOCR(lang="ch")

            # 根据对象拥有的方法精确判断版本（避免误判）
            self._paddleocr_version = 3 if hasattr(self.ocr, "predict") else 2

            return self.ocr

        except ImportError:
            return None
        except Exception as e:
            logger.warning("PaddleOCR init failed: %s", e)
            return None

    def extract_text(self, image_file) -> str:
        """
        从图片中提取文字。

        Args:
            image_file: Streamlit UploadedFile 对象，或文件路径字符串

        Returns:
            识别出的文本字符串
        """
        engine = self.init_engine()
        if engine is None:
            return "OCR 引擎初始化失败，请确认已安装 paddleocr 和 paddlepaddle"

        # 使用唯一文件名避免并发冲突
        if hasattr(image_file, "name"):
            suffix = os.path.splitext(image_file.name)[1] or ".jpg"
            is_path = False
        else:
            suffix = ".jpg"
            is_path = True

        temp_path = os.path.join(self._temp_dir, f"{uuid.uuid4().hex}{suffix}")

        try:
            if is_path:
                # 直接传入文件路径
                ocr_input = image_file
            else:
                # 保存 Streamlit UploadedFile 到临时路径
                with open(temp_path, "wb") as f:
                    f.write(image_file.getbuffer())
                ocr_input = temp_path

            # 执行 OCR 识别（兼容 2.x 和 3.x）
            if self._paddleocr_version >= 3:
                result = engine.predict(ocr_input)
            else:
                result = engine.ocr(ocr_input, cls=True)

            # 解析结果（兼容新旧格式）
            extracted_text = self._parse_result(result)

            if not extracted_text:
                return "OCR 未识别到文字内容，请确认图片中包含清晰文字"

            return "\n".join(extracted_text)

        except Exception as e:
            return f"OCR 识别出错: {type(e).__name__}: {str(e)}"

        finally:
            # 清理临时文件
            try:
                if not is_path and os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass

    def _parse_result(self, result) -> list:
        """
        解析 OCR 结果，兼容 PaddleOCR 2.x 和 3.x 的返回格式。

        PaddleOCR 2.x 格式:
            [[ (bbox, (text, confidence)), ... ], ...]   ← 按图片分组
            每个元素: ([[x1,y1],...], ("文字", 0.95))

        PaddleOCR 3.x 格式:
            [OCRResult(...), ...]   ← 每个图片一个结果对象
            OCRResult 有属性: rec_texts, rec_scores, det_bboxes
            或是一个可迭代对象，每个元素有属性: rec_text, rec_score
        """
        texts = []

        if not result:
            return texts

        for item in result:
            if item is None:
                continue

            # ---- 格式探测：PaddleOCR 2.x ----
            # item 是一个 list，元素为 (bbox, (text, score))
            if isinstance(item, (list, tuple)):
                for sub_item in item:
                    if isinstance(sub_item, (list, tuple)) and len(sub_item) >= 2:
                        # sub_item = (bbox, (text, score))  2.x 格式
                        info = sub_item[1]
                        if isinstance(info, (list, tuple)) and len(info) >= 2:
                            text = info[0]
                            score = info[1]
                            if score > 0.5 and text.strip():
                                texts.append(text.strip())
                        elif isinstance(info, str):
                            if info.strip():
                                texts.append(info.strip())

            # ---- 格式探测：PaddleOCR 3.x 对象 ----
            elif hasattr(item, "rec_texts"):
                # OCRResult 对象，有 rec_texts 和 rec_scores 属性
                item_texts = getattr(item, "rec_texts", [])
                item_scores = getattr(item, "rec_scores", [])
                for i, t in enumerate(item_texts):
                    score = item_scores[i] if i < len(item_scores) else 1.0
                    if score > 0.5 and t.strip():
                        texts.append(t.strip())

            elif hasattr(item, "rec_text"):
                # 单个识别结果对象
                text = getattr(item, "rec_text", "")
                score = getattr(item, "rec_score", 1.0)
                if score > 0.5 and text.strip():
                    texts.append(text.strip())

            # ---- 格式探测：纯字典 ----
            elif isinstance(item, dict):
                if "rec_texts" in item:
                    for t, s in zip(item.get("rec_texts", []),
                                    item.get("rec_scores", [1.0])):
                        if s > 0.5 and t.strip():
                            texts.append(t.strip())
                elif "rec_text" in item:
                    t = item["rec_text"]
                    s = item.get("rec_score", 1.0)
                    if s > 0.5 and t.strip():
                        texts.append(t.strip())
                elif "text" in item:
                    t = item["text"]
                    if t.strip():
                        texts.append(t.strip())

        return texts

    def __del__(self):
        """清理临时目录"""
        try:
            if hasattr(self, "_temp_dir") and os.path.exists(self._temp_dir):
                import shutil
                shutil.rmtree(self._temp_dir, ignore_errors=True)
        except Exception:
            pass
