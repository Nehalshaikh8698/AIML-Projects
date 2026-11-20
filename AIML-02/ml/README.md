# ML workflows

## Chest X-ray classification

`train.py` trains a DenseNet-121 multi-label classifier using NIH ChestX-ray14. It groups images by the patient identifier prefix in `Image Index`, then creates train/validation/test splits. The saved checkpoint is consumed by the FastAPI backend.

## CT segmentation

`train_segmentation.py` trains a MONAI 3D U-Net using Medical Segmentation Decathlon Task06_Lung. It expects `dataset.json`, `imagesTr/`, and `labelsTr/` under `data/Task06_Lung/`. It saves a checkpoint consumed by the separate `/api/ct/segment` endpoint.

These are educational baseline pipelines. Review `docs/model-card.md` and `docs/segmentation-model-card.md`; neither model is clinically validated.

## Low-storage NIH subset

You do not need the complete 42–45 GB NIH collection to validate this pipeline. Download one image archive only, such as `images_001.tar.gz`, and the `Data_Entry_2017_v2020.csv` metadata file. `train.py` ignores CSV rows for image files that are not present in the local image directory. Use `--limit 4000` for a first run on the partial set. Do not claim the subset model is equivalent to a model trained on the complete NIH dataset.
