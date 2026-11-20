# Architecture notes

```text
React (Vite)
  |  /api/* and /media/*
  v
FastAPI
  |-- upload validation + image/DICOM conversion
  |-- DenseNet-121 X-ray inference + Grad-CAM
  |-- MONAI 3D U-Net CT segmentation
  |-- Study and review API
  |-- Orthanc REST client
  |           |
  |           v
  |        Orthanc (DICOM)
  |
  v
MongoDB (studies + append-only audit events)
```

The chest X-ray classification checkpoint is `backend/app/models/densenet121_nih14.pth`; the CT segmentation checkpoint is `backend/app/models/unet3d_msd_lung.pth`. The web dashboard is a thin client. Model loading and image inference happen in the backend. The model checkpoint is mounted into the backend image/container at `backend/app/models/densenet121_nih14.pth`. Static scan previews and Grad-CAM overlays are stored under `backend/storage/` for this local educational build.

For a real environment, add authenticated users, authorization, encrypted object storage, retention policies, structured DICOM validation, monitoring, secure deployment, and clinical/regulatory review. Do not expose the demo ports or credentials to the public internet.
