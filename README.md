________________________________________________________________
# 🫁 IMITATE: Image Registration with Context for 4D-CT Artefact Correction.
________________________________________________________________

This script provides a flexible and modular interface to train IMITATE models for time-frame recovery on 4D-CT data using a wide range of configurable parameters.
It supports different training strategies, loss combinations, and Weights & Biases logging.

________________________________________________________________
# TRAINING IMITATE 

main_train.py orchestrates the training for the different models: both IMITATE and "scaled" registration (e.g VoxelMorph) style training.

## 💡 Example Usage
The results in the paper were obtained with: 

1 - for VoxelMorph style with 4 inputs :
```
python main_train.py
--weight_sim 0.7 --weight_dice 0.3 --weight_reg 0.063 --agreement_weight 0.7
-n 2 -i 4
--no-full_res_training
--fixed_as_input
--csv_paths_train train_csvs
--csv_paths_val train_val
```

2- for VoxelMorph style with 4 inputs + conidtional UNet encoding amplitude signal to dim 32 :
```
python main_train.py
--weight_sim 0.7 --weight_dice 0.3 --weight_reg 0.063 --agreement_weight 0.7
-n 2 -i 4 -t 32
--no-full_res_training
--fixed_as_input
--csv_paths_train train_csvs
--csv_paths_val train_val
```

3- for IMITATE with 4 inputs : 
```
python main_train.py
--weight_sim 0.7 --weight_dice 0.3 --weight_reg 0.063 --agreement_weight 0.7
-n 2 -i 4 -t 32
--no-full_res_training
--no-fixed_as_input
 --csv_paths_train train_csvs
--csv_paths_val train_val
```

## ⚙️ Arguments
Important argument flags are :
<ul>
  <li>  --num_model_inputs (-i) : Number of inputs to DIR network (i.e number of context frames to be registered).  </li>
  <li> --time_encoding_dim (-t) : number of dimensions to ecnode auxiliary signal to. </li>
  <li> --fixed_as_input  : If a fixed reference is used for DIR this is a classic registration paradigm and the new image to sytnhesize will be obtained by scaling the predicted DDF</li>
  <li> --no-fixed_as_input : On the other hand, if no fixed reference is used, we are targeting a certain signal value. This is the novelty of IMITATE. </li>
</ul>

### 🧪 Training Parameters
| Argument | Description | Default |
|---------|-------------|---------|
| `--batch_size`, `-b` | Number of samples per batch | `64` |
| `--learning_rate`, `-l` | Learning rate for optimization | `1e-3` |
| `--weight_decay`, `-w` | Optimizer weight decay  | `0.0` |
| `--optimizer` | Optimizer to use (e.g., `Adam`, `SGD`) | `"Adam"` |
| `--scheduler` | Learning rate scheduler (e.g., `CosineAnnealing`) | `"CosineAnnealing"` |
| `--max_epochs`, `-e` | Number of training epochs | `100` |

### 📐 Loss Weights
| Argument | Description | Default |
|---------|-------------|---------|
| `--weight_sim`, `-s` | Weight for similarity loss | `0.8` |
| `--weight_reg`, `-r` | Weight for regularization loss | `0.04` |
| `--weight_dice`, `-d` | Weight for Dice loss | `0.8` |
| `--agreement_weight` | Weight for inter-prediction agreement | `0.0` |

### 🧩 Model Configuration
| Argument | Description | Default |
|---------|-------------|---------|
| `--model`, `-m` | Model architecture name (e.g., `attention`) | `"attention"` |
| `--num_perumation_train`, `-n` | Number of permutations in training | `5` |
| `--num_model_inputs`, `-i` | Number of input frames per sample | `11` |
| `--time_encoding_dim`, `-t` | Time (auxiliary signal) encoding dimensionality (if used) | `None` |

### 🗂 Data Settings
| Argument | Description | Default |
|---------|-------------|---------|
| `--data_mode` | Ordering of input data (`ordered`, `random`, etc.) | `"ordered"` |
| `--cache_rate` | Rate of MONAI data caching to speed up training | `0.4` |

### ⚙️ Boolean Flags
| Argument | Description |
|---------|-------------|
| `--fixed_as_input` / `--no-fixed_as_input` | Use a fixed frame as reference input. **Setting this to False results in IMITATE** |
| `--full_res_training` / `--no-full_res_training` | Train using full-resolution or half-res inputs |
| `--log_wandb` / `--no-log_wandb` | Log training to Weights & Biases |

### 🧼 Preprocessing Options
| Argument | Description | Default |
|---------|-------------|---------|
| `--detrend` | Whether to detrend amplitude signal data | `"False"` |
| `--work_on_phase` | Whether to use phase signal data instead of amplitudes signal from Vxp file | `"False"` |

### 📁 Data Input
| Argument | Description |
|---------|-------------|
| `--csv_paths_train` | Path to CSV file(s) listing training 4D-CT series |
| `--csv_paths_val` | Path to CSV file(s) listing validation 4D-CT series |

### 🔥 Knowledge Distillation
| Argument | Description | Default |
|---------|-------------|---------|
| `--weight_distillation` | Weight for distillation loss | `0.0` |
| `--teacher_model_name` | Path or name of the teacher model | `"None"` |

## 📝 Notes

- Use the `--log_wandb` flag to track experiments in real-time.

________________________________________________________________
# 🩺  Inference

Use construct_4DCT.py with the saved model from training.

## ✨ Example Usage

Basic inference:
```
python inference.py
--model_path "models/best_model.pth"
--patient_path "data/patient_001.csv"
--target 0.5
```
```
python inference.py
--model_path "models/best_model.pth"
--patient_path "data/patient_001.csv"
--target 0.5
--eval
--save_nifty
--wandb_project "4dct-eval"
```
## ⚙️ Arguments

### 📁 Input Paths
| Argument | Description |
|----------|-------------|
| `--model_path` | Path to the trained model file |
| `--patient_path` | Path to the CSV describing the patient’s 4D-CT data |

### 🎯 Inference Configuration
| Argument | Description | Default |
|----------|-------------|---------|
| `--target` | Target value used for inference (e.g., amplitude or phase) | *(required)* |
| `--threshold` | Threshold used during post-processing or decision-making | `-1` |

### 🧪 Evaluation & Output
| Argument | Description | Default |
|----------|-------------|---------|
| `--eval` / `--no-eval` | Whether to run evaluation metrics | `False` |
| `--save_nifty` / `--no-save_nifty` | Whether to save output predictions as NIfTI files | `False` |

### 📊 Logging
| Argument | Description |
|----------|-------------|
| `--wandb_project`, `-w` | Name of the Weights & Biases project for logging |


Note:
    Conditional Attention Unet is defined in src/conditional_model.py
    A 4D-CT dataset containing amplitude respiration files (VXP) is required for the method. 


