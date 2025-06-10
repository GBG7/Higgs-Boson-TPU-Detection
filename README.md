# Detecting the Higgs Boson File using Kaggles TPUs

by GBG7

---

## Overview

This repository contains a Jupyter Notebook for detecting the Higgs Boson particle using Kaggle's TPUs.

---

## Usage

1. **Upload the `.ipynb` file onto [Kaggle](https://www.kaggle.com/).**
2. Go to `Settings > Accelerator` and select **TPUVM3-8**.

---

## Requirements

- **TensorFlow v2.18.0**

> ⚠️ Note: Various other guides only work for TF 2.10.0.  
> This code has been updated to support the latest version (2.18.0).

---

## Accessing Training Data

Link to the notebook can be found [here](https://www.kaggle.com/code/lilsolar/solar-higgs-boson-tpus)
Training files can be found [here](https://www.kaggle.com/c/higgs-boson/data).

However, you do not need to download the data manually. The following code will handle accessing the data for you:

```python
data_dir = KaggleDatasets().get_gcs_path('higgs-boson-dataset')
print(tf.io.gfile.listdir(data_dir))
```


Writeup:
## ⚛️ Higgs-Boson Detection on Kaggle (TPU)

Short pipeline that trains a **Wide-and-Deep neural network** on the 11 M-event Higgs dataset, end-to-end on a Google TPU v3-8.

| Stage | Details | Stack |
|-------|---------|-------|
| **1. TPU set-up** | Auto-detect TPU, init via `TPUStrategy`; falls back to CPU/GPU. | TensorFlow 2.18 |
| **2. Data loading** | 24 TFRecord shards → `tf.data` → `shuffle-repeat-batch-prefetch` (batch = 8 × 2048).  Custom decoder parses the `features` tensor (28 floats) and expands scalar label to shape *(B, 1)*. | `tf.data`, `FixedLenFeature` |
| **3. Model** | *Wide branch* = single dense logit.<br>*Deep branch* = 5× `Dense→ReLU→Dropout` (128 units).<br>Logits summed; **no sigmoid** (train **from-logits**). | Keras Functional API |
| **4. Compile** | `BinaryCrossentropy(from_logits=True)` + `AUC` / `BinaryAccuracy(threshold=0)`.<br>Adam (1 e-4, `clipnorm=1.0`). | `tf.keras` |
| **5. Training** | `steps_per_epoch = 640`, `validation_steps = 80`.<br>Callbacks: early-stopping, ReduceLROnPlateau. | TPU (8 replicas) |
| **6. Monitoring** | Loss & AUC logged to `history`; plots saved via Matplotlib. | `pandas`, `matplotlib` |

Result: stable training (no NaNs), ~50 s per epoch on TPU, validation **AUC ≈ 0.832** after 3 epochs.

> **Key take-away:** demonstrates full TPU pipeline, raw-logit training for numerical stability, and scalable data input with 11 million events—all wrapped in <50 lines of clear TensorFlow.

