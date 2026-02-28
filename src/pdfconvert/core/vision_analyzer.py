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

    for attempt in range(3):
        resp = requests.post(
            f"{LMS_BASE_URL}/v1/chat/completions",
            json=payload,
            timeout=120,
            proxies={"http": None, "https": None},
        )
        if resp.status_code == 503 and attempt < 2:
            time.sleep(3)
            continue
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return clean_model_output(content)

    raise RuntimeError("LM Studio API 持续返回 503，请检查模型是否已加载")
