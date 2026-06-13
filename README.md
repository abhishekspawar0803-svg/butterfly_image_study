# Butterfly Image Study

A deep learning computer vision project built on a butterfly image dataset, combining **custom CNN classification**, **transfer learning**, and **autoencoder-based image reconstruction** in a single comparative study.

## Project Overview

This project explores multiple deep learning approaches on a fine-grained butterfly image classification task.

The study includes three major components:

- A custom CNN for butterfly species classification
- A transfer learning model for improved classification performance
- An autoencoder for image reconstruction and representation learning

This makes the project more than a standard image classifier — it becomes a broader study of supervised and unsupervised deep learning techniques on the same visual dataset.

## Dataset

The notebook uses the Kaggle dataset:

**Butterfly Image Classification**

The data is accessed using `kagglehub` and includes:

- **6499 labeled training images**
- **2786 test images**
- **75 butterfly species classes**

Example species visible in the notebook include:

- SOUTHERN DOGFACE
- ADONIS
- BROWN SIPROETA
- MONARCH
- APPOLLO
- ATALA

The dataset is organized using image folders plus CSV metadata files:

- `Training_set.csv`
- `Testing_set.csv`

## Part 1: Custom CNN Classification

The first part of the notebook builds a custom convolutional neural network for multi-class butterfly classification.

### Class Distribution

![Class Distribution](images/class_dist.png)

### Classification Accuracy

![Custom CNN Accuracy](images/class_acc.png)

### Classification Loss

![Custom CNN Loss](images/class_loss.png)

### Classification Confusion Matrix

![Custom CNN Confusion Matrix](images/class_conf.png)

This section demonstrates end-to-end image classification using a model built from scratch.

### Custom CNN Results

The custom CNN was trained on **5200 training images** and validated on **1299 validation images** using an 80/20 split from the 6499 labeled butterfly images.

**Dataset and setup**

- Total labeled training images: **6499**
- Test images: **2786**
- Number of classes: **75**
- Input size: **224 × 224 × 3**
- Batch size: **32**

**Model summary**

- Architecture: 4 convolution blocks with BatchNorm + ReLU + MaxPooling, followed by GlobalAveragePooling and dense layers
- Total parameters: **498,699**
- Trainable parameters: **497,739**
- Non-trainable parameters: **960**

**Performance**

- Best validation accuracy during training: **79.83%**
- Best validation loss during training: **0.7880**
- Validation accuracy after evaluation: **78.98%**
- Validation loss after evaluation: **0.7961**

**Classification report summary**

- Macro average precision: **0.80**
- Macro average recall: **0.79**
- Macro average F1-score: **0.79**
- Weighted average precision: **0.81**
- Weighted average recall: **0.79**
- Weighted average F1-score: **0.79**

These results show that the custom CNN can learn fine-grained butterfly species classification reasonably well, even across **75 visually similar classes**, while staying relatively lightweight at under **0.5 million parameters**.

## Part 2: Transfer Learning

The second part of the project applies transfer learning to the same butterfly species classification problem.

### Transfer Learning Accuracy

![Transfer Learning Accuracy](images/trans_acc.png)

### Transfer Learning Loss

![Transfer Learning Loss](images/trans_loss.png)

### Transfer Learning Confusion Matrix

![Transfer Learning Confusion Matrix](images/trans_conf.png)

This helps compare a pretrained feature extractor against the custom CNN baseline.

### Transfer Learning Results

- **Validation Accuracy:** 85.99% (loss: 0.4974)
- **Macro Average F1-score:** 0.85 (weighted avg: 0.87)
- **Per-class highlights:**
  - Perfect classification (F1 = 1.00) on several species (e.g., classes 1, 3, 50, 51, 74)
  - Hardest classes: class 19 (F1 = 0.59), class 38 (F1 = 0.56), class 57 (F1 = 0.52)
- **Metrics:** Accuracy curve, loss curve, confusion matrix, and full classification report (per-class precision, recall, F1 across all 75 species).

> Transfer learning achieved **+6.99% higher validation accuracy** than the custom CNN (85.99% vs 78.98%), confirming the advantage of pretrained feature representations on fine-grained species classification.

## Part 3: Autoencoder Reconstruction

An encoder-decoder convolutional autoencoder was trained on the butterfly dataset for unsupervised representation learning and image reconstruction.

- **Architecture:**
  - **Encoder:** 4× Conv2D blocks (32→64→128→256 filters, MaxPool2D each), output: 14×14×256 latent space
  - **Decoder:** 4× Conv2DTranspose blocks (128→64→32→3 filters, stride 2), output: 224×224×3
  - **Total parameters:** 776,579 (~2.96 MB)
- **Training:** Adam optimizer, MSE loss, 2000 epochs
  - Epoch 1: loss 0.0367 → val_loss 0.0148
  - Epoch 6: loss 0.0070 → val_loss 0.0066 (converging)
- **Evaluation:** Qualitative reconstruction — reconstructed images are visually compared against originals to assess encoder compression quality.

### Original Images

![Original Butterfly Images](images/orig.png)

### Reconstructed Images

![Reconstructed Butterfly Images](images/reconstruct.png)

This section explores unsupervised representation learning and image reconstruction on the butterfly dataset.

## Repository Structure

```bash
butterfly_image_study/
│── butterfly_study.ipynb
│── README.md
│── requirements.txt
│── .gitignore
│── images/
│   ├── class_dist.png
│   ├── class_conf.png
│   ├── class_acc.png
│   ├── class_loss.png
│   ├── orig.png
│   ├── reconstruct.png
│   ├── trans_acc.png
│   ├── trans_loss.png
│   └── trans_conf.png
```

## How to Run

1. Clone the repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open the notebook:

```bash
jupyter notebook butterfly_study.ipynb
```

4. Run all cells to:

- download the dataset,
- preprocess the butterfly images,
- train the custom CNN,
- evaluate transfer learning performance,
- and reconstruct images using the autoencoder.

## Highlights

- Fine-grained image classification across 75 butterfly species
- Comparison between custom CNN and transfer learning approaches
- Autoencoder-based image reconstruction
- TensorFlow / Keras computer vision workflow
- Strong comparative deep learning portfolio project

## Future Improvements

- Add top-k prediction visualization for difficult species pairs
- Compare multiple pretrained backbones for transfer learning
- Use the autoencoder encoder as a feature extractor for downstream classification
- Build a lightweight inference demo for butterfly species prediction
