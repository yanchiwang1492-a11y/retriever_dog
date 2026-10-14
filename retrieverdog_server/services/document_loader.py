"""
文档加载服务
==============
负责将各种格式的文件解析为 LlamaIndex 的 Document 对象列表。

支持的文件类型及解析方式：
1. PDF (.pdf)            → LlamaParse → Markdown 格式 Document
2. 图片 (.png/.jpg/.jpeg/.bmp/.tiff) → pytesseract OCR → 文本 Document
3. Word (.docx)          → python-docx → 文本 Document
4. Excel (.xlsx/.xls)    → pandas → 文本 Document
5. PPT (.pptx)           → python-pptx → 文本 Document
6. 纯文本/代码 (.txt, .md, .csv, .json, .xml, .html, .css, .js, .py, ...) → 直接读取

知识点对应：
- 02.LlamaIndex.md 第 3 节：Document / Node 概念
- 02.LlamaIndex.md 第 5 节：SimpleDirectoryReader 数据加载
- 03.LlamaIndex-RAG.md 第 1 节：LlamaParse PDF 解析
"""

import os
from pathlib import Path
from typing import List

from llama_index.core import Document
from llama_parse import LlamaParse

from config.settings import (
    LLAMA_CLOUD_API_KEY,
    TESSERACT_CMD,
    TESSERACT_LANG,
)


# ============================================================
# 支持的文件扩展名分类
# ============================================================

# PDF 格式
PDF_EXTENSIONS = {".pdf"}

# 图片格式（需要 OCR）
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

# Word 文档
WORD_EXTENSIONS = {".docx"}

# Excel 表格
EXCEL_EXTENSIONS = {".xlsx", ".xls"}

# PPT 演示文稿
PPT_EXTENSIONS = {".pptx"}

# 纯文本类（直接读取文本内容）
# 包括常见的文本、代码、数据格式
TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst",
    ".csv", ".json", ".xml", ".yaml", ".yml",
    ".html", ".htm", ".css",
    ".py", ".js", ".ts", ".java", ".c", ".cpp", ".h",
    ".go", ".rs", ".rb", ".php", ".swift", ".kt",
    ".sh", ".bat", ".sql",
}

# 所有支持的扩展名（用于前端校验和后端判断）
ALL_SUPPORTED_EXTENSIONS = (
    PDF_EXTENSIONS | IMAGE_EXTENSIONS | WORD_EXTENSIONS
    | EXCEL_EXTENSIONS | PPT_EXTENSIONS | TEXT_EXTENSIONS
)


def is_supported_file(filename: str) -> bool:
    """判断文件是否为支持的格式"""
    ext = Path(filename).suffix.lower()
    return ext in ALL_SUPPORTED_EXTENSIONS


# ============================================================
# 主入口函数：根据文件类型选择解析器
# ============================================================

def load_document(file_path: str) -> List[Document]:
    """
    加载并解析文档，返回 Document 对象列表。

    根据文件扩展名自动选择合适的解析方式：
    - PDF → LlamaParse（云端解析，效果最好）
    - 图片 → pytesseract OCR（本地识别）
    - Word/Excel/PPT → 对应库提取文本
    - 纯文本 → 直接读取

    参数:
        file_path: 文件的完整路径

    返回:
        List[Document]: LlamaIndex Document 对象列表
                        （PDF 可能返回多个，其他格式通常返回 1 个）
    """
    ext = Path(file_path).suffix.lower()
    filename = os.path.basename(file_path)

    # 通用元数据：每个 Document 都带上源文件名
    metadata = {"filename": filename, "source": filename}

    # ---------- 1. PDF → LlamaParse ----------
    if ext in PDF_EXTENSIONS:
        return _load_pdf_with_llama_parse(file_path, metadata)

    # ---------- 2. 图片 → OCR ----------
    if ext in IMAGE_EXTENSIONS:
        return _load_image_with_ocr(file_path, metadata)

    # ---------- 3. Word 文档 ----------
    if ext in WORD_EXTENSIONS:
        return _load_word(file_path, metadata)

    # ---------- 4. Excel 表格 ----------
    if ext in EXCEL_EXTENSIONS:
        return _load_excel(file_path, metadata)

    # ---------- 5. PPT 演示文稿 ----------
    if ext in PPT_EXTENSIONS:
        return _load_ppt(file_path, metadata)

    # ---------- 6. 纯文本 / 代码 ----------
    if ext in TEXT_EXTENSIONS:
        return _load_text(file_path, metadata)

    # 不支持的格式（理论上不会走到这里，因为上传时已校验）
    raise ValueError(f"不支持的文件格式: {ext}")


# ============================================================
# 1. PDF 解析：LlamaParse
# ============================================================

def _load_pdf_with_llama_parse(file_path: str, metadata: dict) -> List[Document]:
    """
    使用 LlamaParse 解析 PDF，返回 Markdown 格式的 Document 列表。

    知识点（03.LlamaIndex-RAG.md 第 1 节）：
    - LlamaParse 是 LlamaIndex 官方提供的云端 PDF 解析服务
    - 支持将 PDF 解析为 markdown / text / json 三种格式
    - markdown 格式能保留表格、标题层级、代码块等结构
    - 免费额度：每月 1000 页

    参数:
        file_path: PDF 文件路径
        metadata: 附加到 Document 的元数据

    返回:
        List[Document]: 解析后的文档列表（可能多页）
    """
    # 初始化 LlamaParse 解析器
    parser = LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY,
        result_type="markdown",   # 输出 Markdown 格式，保留结构信息
        num_workers=4,            # 并行 worker 数，提高解析速度
        verbose=True,             # 打印详细日志
        language="ch_sim",        # 文档语言：简体中文
        # 自定义解析指令，告诉解析器如何优化解析效果
        system_prompt="""
        请准确解析文档内容，保留所有表格结构、代码块和层级标题。
        对于数学公式，请尽量保留原始格式。
        中文内容请正确识别简体中文字符。
        """,
    )

    # load_data() 直接返回 Document 对象列表
    documents = parser.load_data(file_path)

    # 给每个 Document 补充元数据（LlamaParse 返回的文档可能元数据不全）
    for doc in documents:
        doc.metadata.update(metadata)
        # 将文件名写入 text_template 会用到的字段，方便检索时追踪来源
        doc.excluded_embed_metadata_keys = []  # 元数据都参与 embedding

    print(f"[DocumentLoader] PDF 解析完成，共 {len(documents)} 页")
    return documents


# ============================================================
# 2. 图片 OCR：pytesseract
# ============================================================

def _load_image_with_ocr(file_path: str, metadata: dict) -> List[Document]:
    """
    使用 pytesseract 对图片进行 OCR 文字识别。

    依赖：
    - pytesseract: Tesseract OCR 的 Python 封装
    - Pillow (PIL): Python 图像处理库
    - Tesseract-OCR: 本机安装的 OCR 引擎（Windows 默认路径见 settings.py）

    参数:
        file_path: 图片文件路径
        metadata: 附加到 Document 的元数据

    返回:
        List[Document]: 包含识别文本的单文档列表
    """
    import pytesseract
    from PIL import Image

    # 设置 Tesseract 可执行文件路径（Windows 需要，Linux/mac 通常在 PATH 里）
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    # 打开图片
    img = Image.open(file_path)

    # 执行 OCR 识别
    # lang 参数：chi_sim=简体中文，eng=英文，多个用 + 连接
    text = pytesseract.image_to_string(img, lang=TESSERACT_LANG)

    # OCR 结果可能有大量空行，做简单清理
    text = text.strip()

    if not text:
        # 识别为空时也要返回一个空文档，避免报错
        text = "（图片中未识别出文字内容）"

    # 构造成 LlamaIndex 的 Document 对象
    doc = Document(text=text, metadata=metadata)

    print(f"[DocumentLoader] 图片 OCR 完成，识别文字 {len(text)} 字")
    return [doc]


# ============================================================
# 3. Word 文档解析
# ============================================================

def _load_word(file_path: str, metadata: dict) -> List[Document]:
    """
    使用 python-docx 提取 Word 文档中的文本内容。

    注意：
    - 只提取段落文本，不提取图片、形状等非文本内容
    - 表格会按行拼接文本

    参数:
        file_path: .docx 文件路径
        metadata: 元数据

    返回:
        List[Document]: 单文档列表
    """
    from docx import Document as DocxDocument

    docx_doc = DocxDocument(file_path)
    paragraphs = []

    # 提取所有段落文本
    for para in docx_doc.paragraphs:
        if para.text.strip():
            paragraphs.append(para.text)

    # 提取表格文本（逐行逐单元格拼接）
    for table in docx_doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                paragraphs.append(row_text)

    text = "\n".join(paragraphs)

    if not text:
        text = "（文档中未提取到文本内容）"

    doc = Document(text=text, metadata=metadata)
    print(f"[DocumentLoader] Word 解析完成，共 {len(paragraphs)} 段，{len(text)} 字")
    return [doc]


# ============================================================
# 4. Excel 表格解析
# ============================================================

def _load_excel(file_path: str, metadata: dict) -> List[Document]:
    """
    使用 pandas 读取 Excel 文件，转为文本格式。

    处理逻辑：
    - 读取所有 sheet
    - 每个 sheet 转为 CSV 格式的文本（保留表格结构）
    - 多个 sheet 之间用分隔符拼接

    参数:
        file_path: .xlsx / .xls 文件路径
        metadata: 元数据

    返回:
        List[Document]: 单文档列表
    """
    import pandas as pd

    # 读取所有 sheet
    xls = pd.ExcelFile(file_path)
    sheet_texts = []

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        # DataFrame 转 CSV 文本（保留表头和数据）
        csv_text = df.to_csv(index=False, na_rep="")
        sheet_texts.append(
            f"=== Sheet: {sheet_name} ===\n{csv_text}"
        )

    text = "\n\n".join(sheet_texts)

    if not text.strip():
        text = "（表格中未提取到数据）"

    doc = Document(text=text, metadata=metadata)
    print(f"[DocumentLoader] Excel 解析完成，共 {len(xls.sheet_names)} 个 sheet")
    return [doc]


# ============================================================
# 5. PPT 演示文稿解析
# ============================================================

def _load_ppt(file_path: str, metadata: dict) -> List[Document]:
    """
    使用 python-pptx 提取 PPT 中的文本内容。

    提取范围：
    - 每张幻灯片中的所有形状文本
    - 备注文本

    参数:
        file_path: .pptx 文件路径
        metadata: 元数据

    返回:
        List[Document]: 单文档列表
    """
    from pptx import Presentation

    prs = Presentation(file_path)
    slides_text = []

    for slide_idx, slide in enumerate(prs.slides, 1):
        slide_texts = []

        # 提取幻灯片中的文本框内容
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                slide_texts.append(shape.text.strip())

        # 提取备注
        if slide.has_notes_slide:
            notes_text = slide.notes_slide.notes_text_frame.text.strip()
            if notes_text:
                slide_texts.append(f"[备注] {notes_text}")

        if slide_texts:
            slides_text.append(
                f"=== 第 {slide_idx} 页 ===\n" + "\n".join(slide_texts)
            )

    text = "\n\n".join(slides_text)

    if not text.strip():
        text = "（PPT 中未提取到文本内容）"

    doc = Document(text=text, metadata=metadata)
    print(f"[DocumentLoader] PPT 解析完成，共 {len(prs.slides)} 页")
    return [doc]


# ============================================================
# 6. 纯文本 / 代码文件解析
# ============================================================

def _load_text(file_path: str, metadata: dict) -> List[Document]:
    """
    直接读取纯文本文件（.txt, .md, .csv, .py, .css, .js 等）。

    尝试多种编码方式读取，避免中文乱码。

    参数:
        file_path: 文本文件路径
        metadata: 元数据

    返回:
        List[Document]: 单文档列表
    """
    # 尝试的编码列表（按优先级排列）
    encodings = ["utf-8", "gbk", "gb2312", "latin-1"]

    text = ""
    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                text = f.read()
            break
        except UnicodeDecodeError:
            continue

    if not text.strip():
        text = "（文件内容为空或无法读取）"

    doc = Document(text=text, metadata=metadata)
    print(f"[DocumentLoader] 文本文件读取完成，{len(text)} 字")
    return [doc]
