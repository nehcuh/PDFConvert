import base64
from io import BytesIO
import requests
from PIL import Image
from ..config import LMS_BASE_URL, MAX_TOKENS, TEMPERATURE, MAX_LONG_EDGE


def image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    buf = BytesIO()
    image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{b64}"


def resize_image(image: Image.Image, max_long_edge: int = MAX_LONG_EDGE) -> Image.Image:
    w, h = image.size
    long_edge = max(w, h)
    if long_edge <= max_long_edge:
        return image
    scale = max_long_edge / long_edge
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def clean_model_output(text: str) -> str:
    """清理模型输出中的思考标签和特殊标记"""
    import re

    # 移除 <think>...</think> 标签及其内容
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    # 移除 <|begin_of_box|> 和 <|end_of_box|> 标记
    text = text.replace('<|begin_of_box|>', '').replace('<|end_of_box|>', '')

    # 检测并移除思考过程
    # 模式1: 如果文本以明显的分析/思考开头，尝试找到实际内容
    lines = text.split('\n')

    # 查找第一个不是思考过程的行
    content_start_idx = 0
    in_thinking = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # 检测思考过程的开始标志
        if re.match(r'^(\*\*\d+\.|The user wants|I need to|Let me|I will|First,|Step \d+|\*\*.*Section.*\*\*:|\*\*Plan|\*\*Strategy|\*\*Formatting|\*\*Drafting|\*\*Refined)', stripped, re.IGNORECASE):
            in_thinking = True
            continue

        # 检测思考过程的结束（实际内容开始）
        # 实际内容通常是：定义、标题、列表项、公式等
        if in_thinking and stripped:
            # 如果这行看起来像实际内容（不是思考过程的一部分）
            if (re.match(r'^(#{1,6}\s|\*\s+\*\*|Def:|Definition:|Frequency:|Period:|[A-Z][a-z]+:|\$\$)', stripped) or
                (not re.search(r'(looking|check|wait|let\'s|I will|should|need to|try to|okay)', stripped, re.IGNORECASE) and
                 len(stripped) > 20 and not stripped.startswith('*'))):
                content_start_idx = i
                break

    # 如果找到了内容开始位置，从那里开始保留
    if content_start_idx > 0:
        text = '\n'.join(lines[content_start_idx:])

    # 移除多余的空行（连续3个以上换行符压缩为2个）
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def analyze_page(image: Image.Image, prompt: str, model_id: str) -> str:
    import time

    resized = resize_image(image)
    data_uri = image_to_base64(resized)

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }

    for attempt in range(5):
        try:
            resp = requests.post(
                f"{LMS_BASE_URL}/v1/chat/completions",
                json=payload,
                timeout=600,
                proxies={"http": None, "https": None},
            )
            if resp.status_code == 503 and attempt < 4:
                print(f"  (模型忙碌，等待 5 秒后重试...)", flush=True)
                time.sleep(5)
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return clean_model_output(content)
        except requests.exceptions.Timeout:
            if attempt < 4:
                print(f"  (请求超时，等待 10 秒后重试...)", flush=True)
                time.sleep(10)
                continue
            raise

    raise RuntimeError("LM Studio API 持续返回 503，请检查模型是否已加载")
