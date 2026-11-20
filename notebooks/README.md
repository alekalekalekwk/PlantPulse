# Notebooks

Training runs on Google Colab (free GPU). Planned notebooks:

| Notebook | Purpose |
| --- | --- |
| `01_explore_data.ipynb` | Class counts, sample images, decide which classes to keep |
| `02_baseline_cnn.ipynb` | Small CNN from scratch, as a reference point |
| `03_finetune.ipynb` | MobileNetV3 and EfficientNet-B0, two-stage fine-tuning |
| `04_evaluate.ipynb` | Both test sets, confusion matrix, Grad-CAM, error analysis |
| `05_export.ipynb` | Export to ONNX (web app) and TFLite, write `models/model_meta.json` |

Save checkpoints to Google Drive so a lost Colab session does not lose training.
