# Data

Datasets are not committed (see `.gitignore`). Download them and place them like this:

```
data/
  raw/
    plantvillage/          one folder per class, e.g. Tomato___Early_blight/
    plantdoc/              PlantDoc tomato classes, renamed to match PlantVillage
  processed/               created by scripts/split_dataset.py
    train/ val/ test/
  field/
    images/                own phone photos (not committed)
    labels.csv             one row per photo (committed)
```

## Sources

| Source | Where | Notes |
| --- | --- | --- |
| PlantVillage | Kaggle | Lab photos, plain background. Check the license before publishing a model. |
| PlantDoc | GitHub | Real-world photos. Class names differ from PlantVillage; map them before training. |
| Field photos | Own phone | Test only. Never used for training or for tuning the threshold. |

## Field photo labels

`field/labels.csv` columns:

| Column | Meaning |
| --- | --- |
| `file` | File name inside `field/images/` |
| `label` | Class name, same spelling as `src/plantpulse/config.py` |
| `label_confidence` | `high`, `medium` or `low`: how sure the labeler is |
| `labeled_by` | Who labeled it (e.g. `self`, `extension officer`) |
| `location` | Garden, market, neighbor, etc. |
| `date` | YYYY-MM-DD |
| `notes` | Blur, lighting, two diseases on one leaf, etc. |

Report field accuracy for all photos and again for `high` confidence labels only.
