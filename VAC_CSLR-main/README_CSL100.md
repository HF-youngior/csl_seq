# VAC_CSLR for CSL100 Dataset

This guide provides instructions on how to use the VAC (Visual Alignment Constraint) model with the CSL100 dataset for continuous sign language recognition.

## Prerequisites

- Python 3.6+
- PyTorch >= 1.8.0
- Other dependencies listed in requirements.txt

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare CSL100 dataset**:
   - Ensure you have the CSL100 dataset in `.avi` format
   - The dataset should be structured as:
     ```
     dataset/csl100/color/
     ├── 000000/
     │   ├── P01_s1_00_0_color.avi
     │   ├── P01_s1_00_1_color.avi
     │   └── ...
     ├── 000001/
     │   └── ...
     └── ...
     ```

3. **Run the preparation script**:
   ```bash
   python prepare_csl100.py
   ```
   This script will:
   - Copy the corpus.txt file to the current directory
   - Generate information files for the dataset
   - Create a gloss dictionary for the CSL100 dataset

## Training

To train the VAC model on CSL100 dataset, run:

```bash
python main.py --work-dir ./work_dir/csl100_baseline --config configs/baseline_csl100.yaml --device 0
```

## Evaluation

To evaluate the trained model, run:

```bash
python main.py --load-weights ./work_dir/csl100_baseline/dev_xx.xx_epochxx_model.pt --phase test --config configs/baseline_csl100.yaml --device 0
```

## Key Components

### Data Loader
- `dataset/dataloader_csl.py`: Custom data loader for CSL100 dataset
  - Directly reads frames from .avi video files
  - Supports frame sampling and augmentation
  - Handles CSL100-specific data structure

### Preprocessing
- `preprocess/csl_preprocess.py`: Preprocessing script for CSL100 dataset
  - Generates information files for train/dev/test splits
  - Creates gloss dictionary from corpus.txt
  - Handles CSL100-specific label format

### Configuration
- `configs/csl100.yaml`: Dataset configuration for CSL100
- `configs/baseline_csl100.yaml`: Model configuration for CSL100

## Notes

- The CSL100 dataset uses character-level labels, while the original VAC model was designed for gloss-level labels
- The data loader has been modified to handle .avi video files directly, eliminating the need for frame extraction
- The model configuration has been adjusted to work with the CSL100 vocabulary size

## References

- [Visual Alignment Constraint for Continuous Sign Language Recognition](https://openaccess.thecvf.com/content/ICCV2021/html/Min_Visual_Alignment_Constraint_for_Continuous_Sign_Language_Recognition_ICCV_2021_paper.html)
- [CSL100 Dataset](https://github.com/CSL-KU/CSL-Dataset)
