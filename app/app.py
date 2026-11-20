"""PlantPulse web app: tomato leaf photo in, likely disease and first steps out.

Needs two files next to each other, by default in ../models/:
  model.onnx       exported classifier
  model_meta.json  {"classes": [...], "image_size": 224,
                    "mean": [r, g, b], "std": [r, g, b]}

The class order in model_meta.json must match the model's output order.
Set PLANTPULSE_MODEL_DIR to load them from somewhere else (e.g. on Spaces).
"""

import json
import os
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("PLANTPULSE_MODEL_DIR", APP_DIR.parent / "models"))
CONFIDENCE_THRESHOLD = float(os.environ.get("PLANTPULSE_THRESHOLD", 0.60))

ADVICE = json.loads((APP_DIR / "advice.json").read_text(encoding="utf-8"))
DISCLAIMER = ADVICE["_disclaimer"]


class Classifier:
    def __init__(self, model_dir: Path):
        import onnxruntime as ort

        meta = json.loads((model_dir / "model_meta.json").read_text(encoding="utf-8"))
        self.classes = meta["classes"]
        self.size = int(meta.get("image_size", 224))
        self.mean = np.array(meta.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        self.std = np.array(meta.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)

        self.session = ort.InferenceSession(str(model_dir / "model.onnx"),
                                            providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        # PyTorch exports are NCHW (channels at index 1), Keras exports NHWC.
        self.channels_first = len(inp.shape) == 4 and inp.shape[1] == 3

    def preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB")
        # Resize the short side, then center-crop to a square.
        w, h = image.size
        scale = self.size / min(w, h)
        image = image.resize((max(self.size, round(w * scale)), max(self.size, round(h * scale))),
                             Image.BILINEAR)
        w, h = image.size
        left, top = (w - self.size) // 2, (h - self.size) // 2
        image = image.crop((left, top, left + self.size, top + self.size))

        x = np.asarray(image, dtype=np.float32) / 255.0
        x = (x - self.mean) / self.std
        if self.channels_first:
            x = x.transpose(2, 0, 1)
        return x[np.newaxis].astype(np.float32)

    def predict(self, image: Image.Image) -> dict:
        logits = self.session.run(None, {self.input_name: self.preprocess(image)})[0][0]
        logits = logits.astype(np.float64)
        # Apply softmax unless the model already outputs probabilities.
        if not (np.all(logits >= 0) and abs(logits.sum() - 1.0) < 1e-3):
            e = np.exp(logits - logits.max())
            logits = e / e.sum()
        return {cls: float(p) for cls, p in zip(self.classes, logits)}


def load_classifier():
    try:
        return Classifier(MODEL_DIR), None
    except FileNotFoundError as err:
        return None, f"Model belum tersedia ({Path(err.filename).name} tidak ditemukan di {MODEL_DIR})."


CLASSIFIER, LOAD_ERROR = load_classifier()


def format_advice(cls: str, confidence: float) -> str:
    info = ADVICE.get(cls)
    if info is None:
        return f"### {cls}\n\nBelum ada saran untuk kelas ini."
    lines = [f"### {info['name']}", f"Keyakinan model: {confidence:.0%}", "", info["description"]]
    if info.get("steps"):
        lines += ["", "**Langkah awal:**"] + [f"{i}. {s}" for i, s in enumerate(info["steps"], 1)]
    if info.get("note"):
        lines += ["", f"_{info['note']}_"]
    return "\n".join(lines)


def diagnose(image):
    if CLASSIFIER is None:
        return None, LOAD_ERROR
    if image is None:
        return None, "Unggah atau ambil foto daun tomat dulu."

    probs = CLASSIFIER.predict(image)
    top_cls = max(probs, key=probs.get)
    top_p = probs[top_cls]
    labels = {ADVICE.get(c, {}).get("name", c): p for c, p in probs.items()}

    if top_p < CONFIDENCE_THRESHOLD:
        text = (
            "### Model belum yakin\n\n"
            f"Keyakinan tertinggi hanya {top_p:.0%}. Coba foto ulang:\n\n"
            "1. Satu daun saja, memenuhi sebagian besar foto.\n"
            "2. Cahaya terang tapi tidak silau, hindari bayangan tajam.\n"
            "3. Pastikan foto fokus dan tidak buram."
        )
    else:
        text = format_advice(top_cls, top_p)
    return labels, f"{text}\n\n---\n{DISCLAIMER}"


with gr.Blocks(title="PlantPulse") as demo:
    gr.Markdown("# PlantPulse\nDeteksi awal penyakit daun tomat dari foto.")
    with gr.Row():
        with gr.Column():
            photo = gr.Image(type="pil", sources=["upload", "webcam"], label="Foto daun tomat")
            submit = gr.Button("Periksa", variant="primary")
        with gr.Column():
            scores = gr.Label(num_top_classes=3, label="Kemungkinan")
            result = gr.Markdown()
    submit.click(diagnose, inputs=photo, outputs=[scores, result])

if __name__ == "__main__":
    demo.launch()
