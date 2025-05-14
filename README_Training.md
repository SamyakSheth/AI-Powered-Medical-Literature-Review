
# DeepFake Image Classification

## Training Instructions

The training is implemented in five separate **Jupyter notebooks**, one for each model architecture used in this project:

- `training_notebook_dinov2.ipynb`
- `training_notebook_vit.ipynb`
- `training_notebook_efficientnet.ipynb`
- `training_notebook_resnet18.ipynb`
- `training_notebook_vgg16.ipynb`

Each notebook:

- Loads and splits the dataset into train and validation sets,
- Applies basic preprocessing to the images,
- Trains the model for up to 10 epochs with early stopping,
- Tracks training/validation metrics and losses,
- Saves the best model to the `models/` directory.

### Dependencies

You can install dependencies and the required packages with:

```bash
pip install -r requirements.txt
```

### Required Directory Structure

Ensure the training data is stored in the following structure:

```
data/
├── train/
│   ├── real/
│   └── fake/
│  
models/
│  (empty file to store the trained models)
```

- The `real/` folder should contain all genuine human images.
- The `fake/` folder should contain all generated images (e.g., StyleGAN, Stable Diffusion, Photoshop, InsightFace, etc.).
- The `models/` folder will contain the best models saved.

### How to Use

1. Open the notebook for the model you wish to train (e.g., `ResNet18_Training.ipynb`).
2. Change the `DATA_DIR` to match the path to the training data directory.
3. Run all cells in the notebook from top to bottom.

The notebook will:

- Automatically detect your device (GPU/MPS/CPU),
- Begin training with early stopping,
- Save the best-performing model checkpoint to `models/`.

> Tip: All required libraries (Torch, Transformers, Datasets, etc.) must be installed in your environment beforehand.



## Testing Instructions

This section explains how to evaluate all five models implemented in this project using a unified testing pipeline. The testing script has been designed to be modular, CPU-compatible, and flexible across multiple models and datasets.

### Supported Models

The following models are evaluated on four deepfake image test sets:

- `resnet18`
- `vgg16`
- `efficientnet_b0`
- `vit`
- `dinov2`

Each model is tested on the following datasets:

- `stylegan`
- `stablediffusion v1.5`
- `photoshop`
- `insightface`

### Folder Structure

Ensure your project directory is structured as follows:

```
project/
│
├── models/                          # Pretrained/fine-tuned model weights (.pth or HuggingFace format)
│   ├── resnet18_best.pth
│   ├── vgg16_best.pth
│   ├── efficientnet_best.pth
│   ├── vit_best_model/              # HuggingFace model folder
│   └── dinov2_best_model/           # HuggingFace model folder
│
├── data/test/                       # Test datasets
│   ├── stylegan/
│   ├── stablediffusion v1.5/
│   ├── photoshop/
│   └── insightface/
│
├── confusion_matrices/             # Output folder for confusion matrix images (auto-generated)
├── test_results.csv                # Output file for all evaluation metrics (auto-generated)
├── test_all.py                     # Main testing script
└── README.md                       # This file
```

### Dependencies

You can install dependencies and the required packages with:

```bash
pip install -r requirements.txt
```

### Running the Test Script

To evaluate **all five models** across **all datasets**, run the following command:

```bash
python test_all.py
```

This will:
- Load each dataset
- Evaluate all models on each dataset
- Save metrics (accuracy, precision, recall, F1-score and confusion matrix) to `test_results.csv`
- Save confusion matrix plots to `confusion_matrices/`

### Output Files

1. **`test_results.csv`**  
   Contains one row per model × domain pair with the following columns:
   ```
   model,domain,accuracy,precision,recall,f1_score,confusion_matrix
   ```

2. **`confusion_matrices/`**  
   Contains `.png` images for each confusion matrix. File naming format:
   ```
   {model}_{domain}_cm.png
   ```

### Notes
- The script runs entirely on CPU.
- If you fine-tune new models, make sure to update the checkpoint paths in `test_all.py`.
