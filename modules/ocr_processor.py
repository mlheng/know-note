# modules/ocr_processor.py
"""
OCR 文字识别处理器
- 优先使用 PaddleOCR（质量最高，但约 500MB）
- 云端回退到 Tesseract OCR（轻量，约 50MB）
- 两款引擎均不可用时给出明确提示
"""
import tempfile
import os
import uuid
import logging

logger = logging.getLogger(__name__)


class OCRProcessor:
    def __init__(self):
        self._engine = None          # "paddle", "tesseract", or None
        self._ocr = None             # OCR 引擎实例
        self._temp_dir = tempfile.mkdtemp(prefix="knownote_ocr_")

    # ------------------------------------------------------------------
    # 引擎探测 & 懒加载
    # ------------------------------------------------------------------
    def _detect_engine(self) -> str:
        """探测可用的 OCR 引擎，返回 "paddle" / "tesseract" / None"""
        if self._engine is not None:
            return self._engine

        # 1. 优先 PaddleOCR（质量最好）
        try:
            from paddleocr import PaddleOCR  # noqa: F401
            self._engine = "paddle"
            return self._engine
        except ImportError:
            pass

        # 2. 回退 Tesseract（轻量）
        try:
            import pytesseract  # noqa: F401
            # 确认 tesseract 二进制可用
            try:
                pytesseract.get_tesseract_version()
            except Exception:
                logger.warning("pytesseract installed but tesseract binary not found")
                self._engine = None
                return None
            self._engine = "tesseract"
            return self._engine
        except ImportError:
            pass

        self._engine = None
        return None

    def _init_paddle(self):
        """初始化 PaddleOCR"""
        if self._ocr is not None:
            return self._ocr
        try:
            from paddleocr import PaddleOCR
            try:
                self._ocr = PaddleOCR(
                    use_angle_cls=True, lang="ch", show_log=False
                )
            except (ValueError, TypeError):
                self._ocr = PaddleOCR(lang="ch")
            return self._ocr
        except Exception as e:
            logger.warning("PaddleOCR init failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def extract_text(self, image_file) -> str:
        """
        从图片中提取文字。

        Args:
            image_file: Streamlit UploadedFile 对象，或文件路径字符串

        Returns:
            识别出的文本字符串（失败时以 "OCR" 开头）
        """
        engine = self._detect_engine()

        if engine == "paddle":
            return self._extract_via_paddle(image_file)
        elif engine == "tesseract":
            return self._extract_via_tesseract(image_file)
        else:
            return (
                "OCR 引擎未安装。\n\n"
                "• 本地运行：pip install paddlepaddle paddleocr （质量最佳）\n"
                "  或 pip install pytesseract （需额外安装 tesseract-ocr）\n"
                "• 云端部署：Streamlit Cloud 免费版推荐 pytesseract"
            )

    # ------------------------------------------------------------------
    # PaddleOCR 分支
    # ------------------------------------------------------------------
    def _extract_via_paddle(self, image_file) -> str:
        engine = self._init_paddle()
        if engine is None:
            return "OCR 引擎初始化失败，请确认已安装 paddleocr 和 paddlepaddle"

        # 处理输入
        if hasattr(image_file, "name"):
            suffix = os.path.splitext(image_file.name)[1] or ".jpg"
            is_path = False
        else:
            suffix = ".jpg"
            is_path = True

        temp_path = os.path.join(self._temp_dir, f"{uuid.uuid4().hex}{suffix}")

        try:
            if is_path:
                ocr_input = image_file
            else:
                with open(temp_path, "wb") as f:
                    f.write(image_file.getbuffer())
                ocr_input = temp_path

            # 兼容 PaddleOCR 2.x / 3.x
            version = 3 if hasattr(engine, "predict") else 2
            if version >= 3:
                result = engine.predict(ocr_input)
            else:
                result = engine.ocr(ocr_input, cls=True)

            texts = self._parse_paddle_result(result)
            if not texts:
                return "OCR 未识别到文字，请确认图片中包含清晰文字"
            return "\n".join(texts)

        except Exception as e:
            return f"OCR 识别出错: {type(e).__name__}: {str(e)}"
        finally:
            try:
                if not is_path and os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Tesseract 分支
    # ------------------------------------------------------------------
    def _extract_via_tesseract(self, image_file) -> str:
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            return "Tesseract 未安装。请执行: pip install pytesseract Pillow"

        # 准备 PIL Image
        if hasattr(image_file, "read"):
            # Streamlit UploadedFile
            img = Image.open(image_file)
        elif isinstance(image_file, str):
            img = Image.open(image_file)
        else:
            return "不支持的图片格式"

        try:
            # 中文 + 英文识别
            text = pytesseract.image_to_string(img, lang="chi_sim+eng")
            text = text.strip()
            if not text:
                return "OCR 未识别到文字，请确认图片中包含清晰文字"
            return text
        except Exception as e:
            msg = str(e)
            if "TesseractNotFound" in str(type(e).__name__) or "tesseract is not installed" in msg:
                return (
                    "Tesseract OCR 系统包未安装。\n\n"
                    "云端部署请在 packages.txt 中添加：\n"
                    "  tesseract-ocr\n"
                    "  tesseract-ocr-chi-sim"
                )
            return f"Tesseract OCR 识别出错: {msg[:200]}"

    # ------------------------------------------------------------------
    # PaddleOCR 结果解析（保留原逻辑）
    # ------------------------------------------------------------------
    def _parse_paddle_result(self, result) -> list:
        texts = []
        if not result:
            return texts

        for item in result:
            if item is None:
                continue

            # PaddleOCR 2.x: list of (bbox, (text, score))
            if isinstance(item, (list, tuple)):
                for sub_item in item:
                    if isinstance(sub_item, (list, tuple)) and len(sub_item) >= 2:
                        info = sub_item[1]
                        if isinstance(info, (list, tuple)) and len(info) >= 2:
                            text, score = info[0], info[1]
                            if score > 0.5 and text.strip():
                                texts.append(text.strip())
                        elif isinstance(info, str) and info.strip():
                            texts.append(info.strip())

            # PaddleOCR 3.x: object with rec_texts / rec_scores
            elif hasattr(item, "rec_texts"):
                item_texts = getattr(item, "rec_texts", [])
                item_scores = getattr(item, "rec_scores", [])
                for i, t in enumerate(item_texts):
                    score = item_scores[i] if i < len(item_scores) else 1.0
                    if score > 0.5 and t.strip():
                        texts.append(t.strip())

            elif hasattr(item, "rec_text"):
                t = getattr(item, "rec_text", "")
                s = getattr(item, "rec_score", 1.0)
                if s > 0.5 and t.strip():
                    texts.append(t.strip())

            elif isinstance(item, dict):
                if "rec_texts" in item:
                    for t, s in zip(item.get("rec_texts", []),
                                    item.get("rec_scores", [1.0])):
                        if s > 0.5 and t.strip():
                            texts.append(t.strip())
                elif "rec_text" in item:
                    t = item["rec_text"]
                    if t.strip():
                        texts.append(t.strip())
                elif "text" in item:
                    t = item["text"]
                    if t.strip():
                        texts.append(t.strip())

        return texts

    def __del__(self):
        try:
            if hasattr(self, "_temp_dir") and os.path.exists(self._temp_dir):
                import shutil
                shutil.rmtree(self._temp_dir, ignore_errors=True)
        except Exception:
            pass
