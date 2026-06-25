# modules/ocr_processor.py
"""
OCR 文字识别处理器
- 优先使用 PaddleOCR（质量最高，但约 500MB）
- 回退到 Tesseract OCR（轻量，需系统包 tesseract-ocr）
- 再回退到 EasyOCR（纯 Python，无需系统包，自动下载模型）
- 内置图片预处理：灰度化、对比度增强、降噪、放大
- 支持多引擎融合投票，提升准确率
"""
import tempfile
import os
import uuid
import logging
import shutil

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)


class OCRProcessor:
    def __init__(self):
        self._engine = None          # "paddle", "tesseract", "easyocr", or None
        self._ocr = None             # OCR 引擎实例
        self._temp_dir = tempfile.mkdtemp(prefix="knownote_ocr_")

    # ------------------------------------------------------------------
    # 图片预处理
    # ------------------------------------------------------------------
    def _load_image(self, image_file) -> Image.Image:
        """统一加载图片，支持 UploadedFile 和文件路径"""
        if hasattr(image_file, "read"):
            image_file.seek(0)
            return Image.open(image_file).convert("RGB")
        elif isinstance(image_file, str):
            return Image.open(image_file).convert("RGB")
        else:
            raise ValueError("不支持的图片格式")

    def _preprocess_for_ocr(self, img: Image.Image) -> list:
        """
        生成多个预处理变体，供多轮 OCR 识别。
        返回 [(label, Image), ...] 列表。
        """
        variants = []

        # 原始图片
        variants.append(("原图", img))

        # 转灰度
        gray = img.convert("L")
        variants.append(("灰度", gray))

        # 灰度 + 对比度增强 1.5x
        enh = ImageEnhance.Contrast(gray)
        contrast = enh.enhance(1.5)
        variants.append(("高对比", contrast))

        # 灰度 + 对比度 2.0x
        contrast2 = enh.enhance(2.0)
        variants.append(("超高对比", contrast2))

        # 灰度 + 锐化
        sharp = gray.filter(ImageFilter.SHARPEN)
        variants.append(("锐化", sharp))

        # 灰度 + 降噪（中值滤波）
        denoised = gray.filter(ImageFilter.MedianFilter(size=3))
        variants.append(("降噪", denoised))

        # 放大处理（小图片放大 2x 再锐化）
        w, h = img.size
        if w < 500 or h < 500:
            upscaled = gray.resize((w * 2, h * 2), Image.LANCZOS)
            upscaled = upscaled.filter(ImageFilter.SHARPEN)
            variants.append(("2x放大", upscaled))

        return variants

    # ------------------------------------------------------------------
    # 引擎探测 & 懒加载
    # ------------------------------------------------------------------
    def _detect_engine(self) -> str:
        """探测可用的 OCR 引擎，返回 "paddle" / "tesseract" / "easyocr" / None"""
        if self._engine is not None:
            return self._engine

        # 1. 优先 PaddleOCR（质量最好）
        try:
            from paddleocr import PaddleOCR  # noqa: F401
            self._engine = "paddle"
            logger.info("OCR engine: paddleocr")
            return self._engine
        except ImportError:
            pass

        # 2. 回退 pytesseract（轻量，需要系统包 tesseract-ocr）
        try:
            import pytesseract  # noqa: F401
            tesseract_bin = self._find_tesseract_binary()
            if tesseract_bin:
                pytesseract.pytesseract.tesseract_cmd = tesseract_bin
                self._engine = "tesseract"
                logger.info("OCR engine: tesseract (binary: %s)", tesseract_bin)
                return self._engine
            else:
                logger.warning("pytesseract installed but tesseract binary not found, trying easyocr...")
        except ImportError:
            pass

        # 3. 回退 EasyOCR（纯 Python，无需系统包，云端最可靠）
        try:
            import easyocr  # noqa: F401
            self._engine = "easyocr"
            logger.info("OCR engine: easyocr")
            return self._engine
        except ImportError:
            pass

        self._engine = None
        return None

    def _find_tesseract_binary(self) -> str | None:
        """在常见路径中定位 tesseract 二进制文件"""
        candidates = [
            "tesseract",
            "/usr/bin/tesseract",
            "/usr/local/bin/tesseract",
            "/opt/homebrew/bin/tesseract",
            "/usr/local/opt/tesseract/bin/tesseract",
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        for path in candidates:
            if shutil.which(path):
                return path
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

    def _init_easyocr(self):
        """初始化 EasyOCR（懒加载，首次调用时下载模型 ~100MB）"""
        if self._ocr is not None:
            return self._ocr
        try:
            import easyocr
            self._ocr = easyocr.Reader(["ch_sim", "en"], gpu=False, verbose=False)
            return self._ocr
        except Exception as e:
            logger.warning("EasyOCR init failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def extract_text(self, image_file) -> str:
        """
        从图片中提取文字。自动预处理 + 多轮融合提升准确率。

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
        elif engine == "easyocr":
            return self._extract_via_easyocr(image_file)
        else:
            return (
                "OCR 引擎未安装，请安装以下任一引擎：\n\n"
                "**方案 A（推荐，轻量）：**\n"
                "```bash\n"
                "pip install pytesseract Pillow\n"
                "sudo apt install tesseract-ocr tesseract-ocr-chi-sim  # Linux\n"
                "```\n"
                "Windows 用户请从 https://github.com/UB-Mannheim/tesseract/wiki 下载安装\n\n"
                "**方案 B（纯 Python，无需系统包）：**\n"
                "```bash\n"
                "pip install easyocr\n"
                "```\n\n"
                "**方案 C（质量最佳，约 500MB）：**\n"
                "```bash\n"
                "pip install paddlepaddle paddleocr\n"
                "```"
            )

    # ------------------------------------------------------------------
    # 通用：多轮预处理 OCR + 结果融合
    # ------------------------------------------------------------------
    def _multi_pass_ocr(self, image_file, ocr_func) -> str:
        """
        对图片的多个预处理变体分别执行 OCR，融合投票选出最佳结果。
        ocr_func(image_array_or_pil) -> list of (text, confidence)
        """
        try:
            img = self._load_image(image_file)
        except ValueError as e:
            return f"OCR 引擎错误：{e}"

        variants = self._preprocess_for_ocr(img)

        # 收集所有识别结果（去重 + 计数投票）
        text_votes = {}       # text -> total confidence
        text_counts = {}      # text -> occurrence count

        for label, variant in variants:
            try:
                variant_array = np.array(variant)
                results = ocr_func(variant_array)
                for item in results:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        t, conf = item[0], item[1]
                    elif isinstance(item, str):
                        t, conf = item, 1.0
                    else:
                        continue
                    t = str(t).strip()
                    if not t:
                        continue
                    text_votes[t] = text_votes.get(t, 0) + conf
                    text_counts[t] = text_counts.get(t, 0) + 1
            except Exception:
                continue

        if not text_votes:
            return "OCR 未识别到文字，请确认图片中包含清晰文字"

        # 按「置信度总和 × 出现次数」排序，过滤只出现 1 次且低置信度的
        scored = []
        for t, total_conf in text_votes.items():
            count = text_counts[t]
            if count >= 2 or total_conf >= 0.6:
                scored.append((t, total_conf))
        scored.sort(key=lambda x: -x[1])

        lines = [t for t, _ in scored]
        return "\n".join(lines) if lines else "OCR 未识别到文字，请确认图片中包含清晰文字"

    # ------------------------------------------------------------------
    # PaddleOCR 分支
    # ------------------------------------------------------------------
    def _extract_via_paddle(self, image_file) -> str:
        engine = self._init_paddle()
        if engine is None:
            return "OCR 引擎初始化失败，请确认已安装 paddleocr 和 paddlepaddle"

        # PaddleOCR 走原有路径（自带预处理）
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
        except ImportError:
            return "OCR 引擎错误：pytesseract 未安装。请执行: pip install pytesseract Pillow"

        tesseract_bin = self._find_tesseract_binary()
        if tesseract_bin:
            pytesseract.pytesseract.tesseract_cmd = tesseract_bin

        def tesseract_ocr(img_array):
            """Tesseract OCR 单次调用，返回 [(text, conf)]"""
            pil_img = Image.fromarray(img_array)
            try:
                data = pytesseract.image_to_data(pil_img, lang="chi_sim+eng",
                                                 output_type=pytesseract.Output.DICT)
                results = []
                for i, txt in enumerate(data["text"]):
                    t = txt.strip()
                    if t:
                        conf = int(data["conf"][i]) / 100.0 if data["conf"][i] != "-1" else 0.5
                        results.append((t, conf))
                return results
            except Exception:
                # fallback: plain image_to_string
                raw = pytesseract.image_to_string(pil_img, lang="chi_sim+eng")
                return [(line.strip(), 0.8) for line in raw.splitlines() if line.strip()]

        return self._multi_pass_ocr(image_file, tesseract_ocr)

    # ------------------------------------------------------------------
    # EasyOCR 分支（纯 Python，无需系统包）
    # ------------------------------------------------------------------
    def _extract_via_easyocr(self, image_file) -> str:
        engine = self._init_easyocr()
        if engine is None:
            return "OCR 引擎错误：EasyOCR 初始化失败，请确认已安装 easyocr"

        def easyocr_ocr(img_array):
            """EasyOCR 单次调用，返回 [(text, conf)]"""
            results = engine.readtext(img_array, detail=1, paragraph=False)
            return [(r[1], r[2]) for r in results if r[2] > 0.3 and str(r[1]).strip()]

        return self._multi_pass_ocr(image_file, easyocr_ocr)

    # ------------------------------------------------------------------
    # PaddleOCR 结果解析
    # ------------------------------------------------------------------
    def _parse_paddle_result(self, result) -> list:
        texts = []
        if not result:
            return texts

        for item in result:
            if item is None:
                continue

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
                shutil.rmtree(self._temp_dir, ignore_errors=True)
        except Exception:
            pass
