# Model Card — DenseNet-121 / NIH ChestX-ray14

## Intended use

Educational experimentation with multi-label classification of frontal chest radiographs and an accompanying Grad-CAM visualization. This model is not intended for clinical diagnosis, triage, treatment decisions, or autonomous clinical workflows.

## Model details

- Architecture: Torchvision DenseNet-121 with a 14-output linear classification head.
- Learning task: multi-label classification with sigmoid probabilities.
- Training data: NIH ChestX-ray14, subject to its dataset terms and known label limitations.
- Labels: Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening, Hernia.
- Input: 224 × 224 RGB representation of a chest radiograph, normalized with ImageNet channel statistics.
- Explanation: Grad-CAM from the final DenseNet dense block.

## Important limitations

- The NIH labels were mined from radiology reports and contain uncertainty/noise.
- A seeded random image-level split can leak patient-specific information across splits. Use patient-level splits for rigorous research.
- Performance can vary with institution, scanner, portable AP vs PA view, image quality, patient population, and prevalence.
- The 0.50 threshold is a display default, not an optimized or calibrated clinical operating point.
- A heatmap is not a causal explanation, lesion segmentation, or proof of correctness.
- The model does not establish the presence or absence of disease in an individual.
- This project has no external validation, clinical evaluation, or regulatory clearance.

## Bias and risk considerations

Medical imaging datasets can underrepresent populations and acquisition settings. Reported findings can correlate with hospital-specific markers, devices, positioning, and image acquisition artifacts. A model may exploit shortcuts. Evaluate subgroup performance and image-source effects before making scientific claims. Never use this application to make or influence patient-care decisions.

## Training and evaluation

The training script reports per-label AUROC and macro AUROC where computable. Metrics depend on the exact split, preprocessing, random seed, hardware, and training duration. Record the generated `metrics.json`, code version, dataset version, and checkpoint hash for each experiment.

## Model status in this repository

No trained checkpoint is distributed. Train the model locally on the permitted NIH dataset. The backend displays `model_loaded=false` and rejects inference until a checkpoint is present, rather than generating misleading results from randomly initialized weights.


## Demo-mode distinction

When `DEMO_MODE=true` and no trained checkpoint is present, the backend generates deterministic illustrative scores from simple image statistics and a visual image map. This mode does not load or execute a neural network and does not generate Grad-CAM. It exists only to demonstrate upload, review, and audit workflows. Never describe demo-mode scores as AI predictions.
