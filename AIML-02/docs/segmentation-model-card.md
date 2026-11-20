# Model Card — MONAI 3D U-Net / MSD Task06_Lung

## Intended use

Educational experimentation with 3D CT segmentation using the Medical Segmentation Decathlon Task06_Lung dataset. It is not intended for clinical diagnosis, treatment planning, radiotherapy planning, or autonomous clinical workflows.

## Model details

- Architecture: MONAI 3D U-Net, one input channel and two output classes.
- Input: NIfTI 3D CT volume, oriented to RAS, resampled to 1.5 × 1.5 × 2.0 mm, and clipped/scaled using a starter CT window of -1000 to 400 HU.
- Output: foreground/background mask using argmax over two output channels.
- Training data: Medical Segmentation Decathlon Task06_Lung; verify the source dataset label definitions and terms before use.
- Preview: the API returns only the middle axial slice for the demo dashboard.

## Limitations

- The provided pipeline is an educational baseline and requires local training; no checkpoint is distributed.
- The validation split is a seeded case-level split of the supplied training list; inspect and document patient/source leakage before research-grade claims.
- The segmentation target is dataset-specific and must not be interpreted as a general lung, tumor, or pathology detector outside the dataset's annotation definition.
- A segmentation mask is not a diagnosis and has no clinical operating threshold.
- Image spacing, orientation, reconstruction kernel, scanner, contrast phase, and acquisition protocol can affect results.
- This model has no external validation, clinical evaluation, or regulatory clearance.

## Risk and bias

The training cohort may not represent other patient populations, scanners, institutions, or acquisition protocols. Segmentation annotations may have inter-reader variability. Evaluate performance on independent data and inspect failure cases before making scientific claims. Never use this model for patient-care decisions.
