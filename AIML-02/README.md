# MedScan AI — AI-Assisted Medical Imaging Review (Educational MVP)

A full-stack educational project for chest X-ray and CT research workflows. It includes a PyTorch DenseNet-121 multi-label classification pipeline, a MONAI 3D U-Net segmentation pipeline, Grad-CAM visual explanations, a FastAPI service, a React review dashboard, MongoDB-backed study/audit records, and optional Orthanc DICOM integration.

> **Important medical notice:** This software is for education and research only. It is not a medical device, is not clinically validated, and must not be used to diagnose, triage, monitor, or treat patients. Predictions and heatmaps are experimental model outputs, not medical advice. Use only properly de-identified research data. A qualified clinician must not rely on this software for patient care.

## What is included

- **ML pipeline:** train/evaluate a DenseNet-121 on NIH ChestX-ray14 labels (14-label multi-label classification).
- **Inference API:** with trained checkpoints installed, accepts chest X-ray PNG/JPEG or DICOM, returns 14 model scores, and produces a Grad-CAM overlay. Before training, optional `DEMO_MODE` provides clearly-labelled simulated outputs only so the upload/review/audit workflow can be shown. A separate CT endpoint accepts NIfTI volumes and produces a 3D U-Net segmentation mask preview when a trained MONAI checkpoint is available.
- **Review dashboard:** upload a scan, inspect model output and heatmap, record an approve/reject review, and inspect the audit trail.
- **Data/model versioning:** DVC pipeline configuration and instructions.
- **DICOM workflow:** Orthanc server plus an API route that fetches the first instance in an Orthanc study and analyzes it. **CT segmentation workflow:** MONAI 3D U-Net trained with Medical Segmentation Decathlon Task06_Lung.
- **Containerization:** Docker Compose for MongoDB, Orthanc, FastAPI, and Vite/React.

## Chosen dataset and task

The classification pipeline uses **NIH ChestX-ray14**. The model predicts 14 reported findings: Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening, and Hernia. It is a multi-label classification task; several labels may be scored for one image.

Dataset information and access: <https://nihcc.app.box.com/v/ChestXray-Nihcc>

The 3D segmentation pipeline uses **Medical Segmentation Decathlon Task06_Lung**. Download the official dataset from <http://medicaldecathlon.com/> and extract it as `data/Task06_Lung/` with `dataset.json`, `imagesTr/`, and `labelsTr/`. Check the dataset documentation for the exact label definitions before interpreting masks.

### Low-storage download (recommended for a laptop)

**Do not download all 12 NIH image archives.** The complete dataset is about 42–45 GB. The image archives are split into batches. You can start with only `images_001.tar.gz` (reported at about 1.9 GB and approximately 4,999 images in one published archive inventory), plus the small metadata CSV. The NIH metadata CSV covers the full dataset; the training loader automatically skips rows whose image files are not present locally. This lets the same 14-label training pipeline work on a partial local subset. Archive sizes can vary by mirror/version, so check the size shown by Box before downloading.

1. In the NIH Box repository, open `CXR8` → `images`.
2. Download only `images_001.tar.gz`. Do not select the other image archives.
3. Download `Data_Entry_2017_v2020.csv` from the `CXR8` folder.
4. Extract the archive so the `.png` files are directly inside `data/NIH-ChestXray14/images/` (not buried in an extra nested folder).
5. After confirming the PNG files extracted correctly, delete the downloaded `.tar.gz` archive to reclaim space.

Expected layout:

```text
data/NIH-ChestXray14/
├── Data_Entry_2017_v2020.csv
└── images/
    ├── 00000001_000.png
    ├── 00000001_001.png
    └── ... (only the downloaded batch)
```

The CSV must contain `Image Index` and `Finding Labels` columns. The loader filters the CSV to image files that actually exist in `images/`. For a first local run, use `--limit 4000` to cap the number of available images used. Do not commit the images to Git.

If even one archive does not fit, do not download it: use a cloud notebook with temporary storage or switch to a smaller sample dataset and adjust the label mapping before training. The CT segmentation dataset is a separate download; skip that phase until the X-ray workflow is working.

## Fastest low-storage setup on Windows (without Docker)

This mode starts the React dashboard and FastAPI backend without downloading NIH images, PyTorch, MONAI, MongoDB, or Orthanc. It uses the clearly-labelled demo workflow and temporary in-memory study storage. DICOM upload decoding is available through the lightweight `pydicom` dependency; Orthanc and persistent MongoDB are optional Docker services.

**Terminal 1 — backend (PowerShell, from the project root):**

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:DEMO_MODE="true"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 — frontend (from the project root):**

```powershell
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. The dashboard supports upload, demo-only visualization, approve/reject review, and audit history. Because this local lightweight setup does not run MongoDB, its records are cleared when the backend restarts. Use Docker Compose for persistent MongoDB and Orthanc services.

## Quick start — run the dashboard immediately (no dataset required)

Prerequisites: Docker Desktop (or Docker Engine + Compose). The first Docker build downloads the service images and Python packages, so it still needs an internet connection and several GB of temporary disk space. It does **not** download the 42 GB NIH dataset.

1. Copy configuration:

   ```bash
   cp .env.example .env
   ```

   On Windows PowerShell:

   ```powershell
   Copy-Item .env.example .env
   ```

2. Leave `DEMO_MODE=true` in `.env` for the first run. No dataset or checkpoint is needed for the dashboard and doctor-review workflow.
3. Start the platform:

   ```bash
   docker compose up --build
   ```

4. Open:
   - React dashboard: <http://localhost:5173>
   - FastAPI / Swagger: <http://localhost:8000/docs>
   - Orthanc Explorer: <http://localhost:8042>

   Orthanc demo login: `demo` / `demo-change-me`. The copied `.env` configures the backend client credentials; to change the Orthanc login itself, also update `RegisteredUsers` in `orthanc/orthanc.json` and keep the backend username/password values in `.env` aligned. Never expose this local-demo configuration to the public internet.

5. Upload a sample PNG/JPG X-ray image to exercise the study queue, image viewer, review decisions, and audit log. With no trained checkpoint and `DEMO_MODE=true`, the backend returns **simulated image-statistics scores and an illustrative image map**. These are not model predictions and the map is not Grad-CAM. The interface labels these outputs as demo-only.

6. To switch to real model inference, train DenseNet121 using the steps below. Once `backend/app/models/densenet121_nih14.pth` exists, the backend automatically prefers that checkpoint over demo mode. Set `DEMO_MODE=false` to disable the fallback entirely.

## Train the model

Create a virtual environment and install ML dependencies. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r ml/requirements.txt
```

Run training:

```bash
python ml/train.py \
  --data-dir data/NIH-ChestXray14/images \
  --labels-csv data/NIH-ChestXray14/Data_Entry_2017_v2020.csv \
  --epochs 3 \
  --batch-size 4 \
  --limit 4000
```

PowerShell equivalent (single line):

```powershell
python ml/train.py --data-dir data/NIH-ChestXray14/images --labels-csv data/NIH-ChestXray14/Data_Entry_2017_v2020.csv --epochs 3 --batch-size 4 --limit 4000
```

Training defaults to CUDA when available and otherwise uses CPU. On a smaller GPU, lower `--batch-size` to `4` or `8`. For a first smoke test, use `--epochs 1 --limit 2000`; this is only a pipeline check, not a useful model. Training downloads ImageNet initialization weights from Torchvision on first run; an internet connection is needed for that download. The trained checkpoint and JSON metrics are written to `backend/app/models/` and `ml/outputs/`.

For the CT model, install the optional MONAI dependencies and run:

```bash
pip install -r ml/requirements-segmentation.txt
python ml/train_segmentation.py --data-dir data/Task06_Lung --epochs 100 --batch-size 1
```

The 3D segmentation training is compute-intensive. Install `ml/requirements-segmentation.txt` first. Begin with `--max-train-cases 4 --epochs 1` to test the pipeline, not to produce a useful model. It writes `backend/app/models/unet3d_msd_lung.pth` and `ml/outputs/segmentation_metrics.json`. The full CT segmentation path needs the separate Medical Segmentation Decathlon Task06_Lung download and is not part of the no-dataset quick demo.

For real classification inference in a local backend environment, install `backend/requirements-ml.txt` on top of `backend/requirements.txt` (choose the PyTorch wheel matching your CPU/GPU from <https://pytorch.org/get-started/locally/>). For trained CT inference, install `backend/requirements-segmentation.txt`. These packages are intentionally excluded from the lightweight demo backend image. For Docker-based real inference, add the relevant optional requirements to `backend/Dockerfile` before rebuilding the backend image.

### Training data notes

- NIH ChestX-ray14 labels are mined from report text and are noisy labels, not definitive diagnoses.
- The classification script groups NIH images by the patient identifier prefix in the filename before splitting into train/validation/test. Verify the split against the original patient metadata before making research-grade claims.
- The default training setup is a practical starter. It is not a validated clinical model; tune thresholds, calibration, and evaluation protocols before making research claims.
- Metrics include per-label AUROC where it can be computed and macro AUROC over valid labels. AUROC is not a measure of clinical safety.

## Run without Docker (optional)

### Backend

```bash
cd backend
python -m venv .venv
# activate the environment, then:
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Set `MONGO_URI` to a reachable MongoDB instance. If MongoDB is unavailable, the app logs a warning and uses an in-memory store; records will be lost on restart. Set `MODEL_PATH` to the trained checkpoint path. The default relative path is `app/models/densenet121_nih14.pth` when launched from `backend/`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Vite proxies `/api` and `/media` to `http://localhost:8000`.

## API overview

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | API, database, and model readiness |
| `GET` | `/api/studies` | List recent studies |
| `GET` | `/api/studies/{study_id}` | Get one study and prediction |
| `POST` | `/api/studies/upload` | Upload chest X-ray PNG/JPEG/DICOM and run classification + Grad-CAM |
| `POST` | `/api/ct/segment` | Upload NIfTI CT volume and run MONAI 3D U-Net segmentation |
| `POST` | `/api/studies/{study_id}/review` | Record `approved` or `rejected` review |
| `GET` | `/api/studies/{study_id}/audit` | Retrieve audit events |
| `GET` | `/api/orthanc/studies` | List studies currently in Orthanc |
| `POST` | `/api/orthanc/studies/{orthanc_id}/analyze` | Fetch first DICOM instance and run inference |

Open `/docs` for the interactive API documentation. The upload API accepts multipart fields `file`, `patient_ref` (use a random non-identifying study code), and `notes`.

## DICOM / Orthanc

Orthanc runs on HTTP `8042` and DICOM `4242`. In the Orthanc Explorer, sign in with the local-demo credentials, then upload a de-identified DICOM study. In the dashboard, use **DICOM studies** to list studies and analyze one. The demo integration uses the first instance in a study; production systems must handle all series/instances, modalities, pixel spacing, transfer syntaxes, and study-level validation. For the 3D route, upload a de-identified `.nii` or `.nii.gz` CT volume using the **Segment CT volume** panel. The interface previews the middle slice and mask only; the full source volume is not exposed in the viewer.

Do not upload real patient-identifiable data into this educational demo. Orthanc and the API are configured for local development and are not hardened for production.

## DVC (dataset and model versioning)

Install DVC:

```bash
pip install dvc
```

Initialize and add the large dataset (after arranging the folder):

```bash
dvc init
dvc add data/NIH-ChestXray14
```

Configure a remote suitable for your team (for example, an S3-compatible bucket or shared storage):

```bash
dvc remote add -d storage <YOUR_DVC_REMOTE_URL>
dvc push
```

The DVC remote is intentionally not preconfigured. Never commit credentials. Track both datasets with DVC (`dvc add data/NIH-ChestXray14` and `dvc add data/Task06_Lung`). Run only the stage you have data for: `dvc repro train` for classification or `dvc repro train_segmentation` for CT segmentation. Run `dvc repro` only after both datasets are available.

## Project structure

```text
medscan-ai/
├── backend/                  # FastAPI service, inference, Grad-CAM, MongoDB, Orthanc
├── frontend/                 # React + Vite radiology review dashboard
├── ml/                       # NIH classification + MSD CT segmentation training, metrics, DVC pipeline
├── orthanc/                  # Orthanc configuration
├── docs/                     # Model card and safety notes
├── tests/                    # Backend tests
├── docker-compose.yml
└── README.md
```

## Known limitations / next improvements

1. No authentication or role-based access control is included. Reviewer identity is entered in the UI for the demo and is not verified.
2. No clinical validation, external validation, prospective evaluation, or regulatory approval is included.
3. The DenseNet model is trained on chest radiographs only; the separate 3D U-Net is trained on MSD Task06_Lung CT volumes. Neither model should be used on unsupported modalities or datasets.
4. Grad-CAM is a coarse saliency visualization, not proof that the model reasoned correctly or localized disease.
5. The upload path stores local image files; production storage needs encryption, retention controls, access checks, and secure audit trails.
6. Orthanc integration analyzes only the first DICOM instance; production code must support full studies and multi-frame DICOM. CT NIfTI inference previews only the middle slice, not a validated full-volume viewer.
7. The dataset may contain demographic and acquisition biases. Review `docs/model-card.md` before interpreting any output.

## License

This repository scaffold is provided for educational use. Check the NIH dataset terms and every pretrained-weight license separately before redistribution or use.
