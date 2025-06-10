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

## ⚛️ Higgs-Boson Detection on TPU (Wide-&-Deep Model)

**One-cell notebook** that trains a Wide-and-Deep neural network on the Higgs-Boson  
TFRecord dataset using a Google Cloud TPU v3-8 inside Kaggle.

| Stage | Highlights | Key APIs |
|-------|------------|----------|
| **Hardware auto-detect** | Falls back to CPU/GPU if TPU not found. | `TPUClusterResolver`, `TPUStrategy` |
| **TFRecord pipeline** | · Parses 28-float feature tensor + label<br>· Expands label to `(batch,1)`<br>· Shuffles \(2¹⁹\), repeats, batches, prefetches. | `tf.data`, `tf.io.parse_tensor` |
| **Model** | **Wide branch** = single dense unit.<br>**Deep branch** = 5 `dense ➜ ReLU ➜ dropout` blocks.<br>Branches added → **raw logits** (no sigmoid). | `keras.Model`, `layers.Add` |
| **Compile** | Adam (1e-4 LR, gradient clip 1.0).<br>`BinaryCrossentropy(from_logits=True)` + `AUC`/`BinaryAccuracy` on logits. | `model.compile` |
| **Training** | • Global batch = 8 × 1024 (≈ 8 k obs)<br>• `steps_per_epoch = 640`, `validation_steps = 80`<br>• `EarlyStopping`, `ReduceLROnPlateau`. | `model.fit` |
| **Results** | Validation loss ≈ 0.50, AUC ≈ 0.83 after 3 epochs (no hyper-tuning). | `history.history` → plotted with Matplotlib |

**Why it matters**

* Demonstrates end-to-end TPU workflow (data → model → plots) in TensorFlow 2.x.  
* Uses *raw-logit training* for numerical stability at very large batch sizes.  
* Fully reproducible: all preprocessing in `tf.data`; random seeds fixed; single requirements line 👉 `tensorflow~=2.15`.

Run in Kaggle:

```bash
# Inside Kaggle notebook
!pip install -q xgboost  # <- only extra dep if not pre-installed
# just press “Run All” – will auto-attach the TPU and finish in ~3 min

