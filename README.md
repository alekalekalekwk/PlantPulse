# PlantPulse

Tomato leaf disease detection from a phone photo, with first-step care advice in Indonesian.

PlantPulse is built for small farmers and home gardeners who cannot quickly reach an agricultural extension officer. A user takes a photo of a tomato leaf, the app predicts the most likely disease, and it shows what to check and do next. When the model is not confident, the app asks for another photo instead of guessing.

> Status: project setup. No model has been trained yet.

## Why this project

Models trained on lab photos (plain backgrounds, good lighting) often score very well on their own test set and then drop sharply on real field photos. PlantPulse measures that gap directly: the headline number is accuracy on photos collected in real gardens and markets, not on the public dataset.

## Goals

| Goal | Target |
| --- | --- |
| Accuracy on the public test split | >= 95% |
| Accuracy on self-collected field photos | >= 75% (the number that matters most) |
| Prediction time on phone / browser | < 1 second per photo |
| Model size | < 20 MB, so it can run offline later |
| Real user testing | At least 5 people try it and give feedback |

## Scope

Version 1 covers **tomato only**. Out of scope: insect pest diagnosis, specific pesticide doses, and any claim of agronomic accuracy. The app gives an early indication and recommends confirming with an extension officer.

## Data

| Source | Contents | Used for |
| --- | --- | --- |
| PlantVillage | Tomato leaves, healthy + 9 diseases, plain lab background | Main training |
| PlantDoc | Leaves photographed in real conditions | Extra training + testing |
| Own field photos | 200-300 phone photos of tomato leaves | Field test only, never trained on |

Public data is split 70 / 15 / 15 (train / val / test). Field photos are kept separate. See [data/README.md](data/README.md) for the folder layout.

## Repository layout

```
app/                Gradio web app (deployable to Hugging Face Spaces)
  app.py
  advice.json       Disease descriptions and first steps (Indonesian)
  requirements.txt  App-only dependencies
data/               Datasets (not committed, see data/README.md)
models/             Exported models and labels (not committed)
notebooks/          Training notebooks (run on Google Colab)
reports/            Figures and evaluation results
scripts/
  split_dataset.py  Builds the train / val / test split
src/plantpulse/
  config.py         Classes, image size, paths, confidence threshold
```

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Split the public dataset after placing it under `data/raw/plantvillage/`:

```bash
python scripts/split_dataset.py --src data/raw/plantvillage --dst data/processed
```

Run the app locally (needs `models/model.onnx` and `models/model_meta.json`, see [app/app.py](app/app.py)):

```bash
python app/app.py
```

## Models compared

1. Small CNN trained from scratch (baseline)
2. MobileNetV3, fine-tuned from ImageNet
3. EfficientNet-B0, fine-tuned from ImageNet

The final model is exported to ONNX for the web app and to TFLite for a possible offline Android version.

## Evaluation

- Accuracy and per-class F1 on both test sets
- Confusion matrix
- Lab vs field accuracy gap
- Prediction time and model size
- Error analysis of 20 wrong predictions, grouped by cause
- Short user feedback form

## Results

To be filled in after training.

## Limitations

To be filled in after evaluation.

## License

Code license to be decided. Check each dataset's license before publishing a trained model.
