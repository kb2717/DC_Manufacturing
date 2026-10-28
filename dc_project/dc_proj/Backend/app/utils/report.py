import base64
import logging
import os
from io import BytesIO

from openai import OpenAI

from app.config import load_config

SYSTEM_PROMPT = (
    "You are an expert car damage assessor working for an insurance company. "
    "Your job is to assess vehicle damage from images, identify damage types and "
    "locations, estimate repair complexity and cost range, and determine if the "
    "claim is valid. Be precise and professional."
)

DEFAULT_MODEL = "Qwen/Qwen3.5-397B-A17B:novita"

logger = logging.getLogger(__name__)

# Shown in the damage summary box when the written report can't be generated.
LIMIT_REACHED = "Severity analysis unavailable because model usage limit reached."
UNAVAILABLE = "Severity analysis unavailable."


def load_vlm(config=None):
    """Build the HF-router client. HF_TOKEN wins; config.yaml is the fallback."""
    api_key = os.getenv("HF_TOKEN")

    if not api_key:
        config = config if config is not None else load_config()
        api_key = (config.get("api") or {}).get("qwen")

    if not api_key or api_key.startswith("YOUR_"):
        # No key is not a fatal error. The app still starts and still detects damage — it just can't write the report.
        logger.warning(
            "No Hugging Face router key found — severity reports are disabled. "
            "Set HF_TOKEN in Backend/.env, or api.qwen in Backend/config.yaml."
        )
        return None

    return OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=api_key,
    )


# 402 = monthly credits depleted, 429 = rate limited. Both mean "out of quota".
_QUOTA_STATUS = (402, 429)


def _failure_message(exc) -> str:
    """A used-up quota comes back on its own, so it gets its own message."""
    if getattr(exc, "status_code", None) in _QUOTA_STATUS:
        return LIMIT_REACHED
    if type(exc).__name__ == "RateLimitError":
        return LIMIT_REACHED
    return UNAVAILABLE


def describe(seg_img, damage_list, vlm, model: str = DEFAULT_MODEL) -> str:
    """Write the damage report from the segmented image and detected classes.

    Never raises. If the model can't be reached, it returns a short message to
    show instead, so the rest of the results still reach the user.
    """
    if vlm is None:
        return UNAVAILABLE

    img_data = _encode_image(seg_img)
    damage_classes = [d["class"] for d in damage_list]

    try:
        response = vlm.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"The following damage types were detected: {damage_classes}. "
                                "Analyse the segmented image and provide: Damage location and "
                                "type, Severity, Estimated repair complexity, a summary of the "
                                "damages."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_data}"},
                        },
                    ],
                },
            ],
        )
    except Exception as exc:
        logger.warning("Report generation failed: %s: %s", type(exc).__name__, exc)
        return _failure_message(exc)

    return response.choices[0].message.content or UNAVAILABLE


def _encode_image(image) -> str:
    buf = BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")