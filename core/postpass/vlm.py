"""Optional VLM scene analysis via Ollama (helper for the operator, not
part of the core pipeline)."""
import base64
from typing import List

import cv2
import numpy as np


def vlm_ollama(video_path: str, start_s: float, fps: float,
               model: str = "qhuihui-qwen3.5-abliterated-q6_k:27b",
               n_samples: int = 8,
               zone: str = "pelvis/groin zone") -> List[str]:
    try:
        import ollama
    except ImportError:
        print("[vlm] `pip install ollama` - not installed, skipping.")
        return []
    cap = cv2.VideoCapture(video_path)
    # Use the reliable start seek (see core.tracking.seek): the POS_FRAMES
    # idiom is unreliable on some containers.  We do NOT consume the first
    # frame here (the linspace loop below re-seeks anyway); just advance the
    # cap to its starting point.
    from core.tracking.seek import seek_to_time_exact
    frame0, _t0, _seek_ok = seek_to_time_exact(cap, start_s, fps=fps)
    if not _seek_ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * fps))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(total - int(start_s * fps), 1), n_samples, dtype=int)
    images_b64 = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * fps) + int(i))
        ret, frame = cap.read()
        if ret:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            images_b64.append(base64.b64encode(buf.tobytes()).decode("utf-8"))
    cap.release()
    prompt = (
        f"Here are {len(images_b64)} frames from a video, in chronological order. "
        f"For EACH frame, state the position of the ACTIVE partner's {zone} "
        f"on the thrusting arc: FORWARD(0%), BACKWARD(100%), MID(50%), or intermediate. "
        f"Also note: pause or active movement? Format per frame: "
        f"Frame N: XX% - [moving forward / moving backward / paused]. Be concise."
    )
    print(f"[vlm] Ollama: {model}, {len(images_b64)} frames...")
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt, "images": images_b64}],
            options={"temperature": 0.1, "num_ctx": 8192},
        )
        text = response["message"]["content"]
        print(f"[vlm] Answer:\n{text}\n")
        return text.split("\n")
    except Exception as e:
        print(f"[vlm] Ollama error: {e}")
        return []
