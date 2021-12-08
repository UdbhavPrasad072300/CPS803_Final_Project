# CPS803_Final_Project (Group 27)

For our project: Effects of Knowledge Distillation on Vision Transformers with CNN Teachers for Small Datasets

## Contributors

Udbhav Prasad <br>
Matthew Meszaros <br>
Jordan Quan <br>
Muhammad Siddiqui <br>

## How to Run

```bash
python <Script to Run>
```

## Installing Requirements:

Install PyTorch from: https://pytorch.org/get-started/locally/

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
```

```bash
pip install matplotlib
```

```
├── README.md               <- README for developers using this project.
├── data                    <- Data used for this project
├── trained_models          <- Trained and serialized models
├── report                  <- Output Results and Report
├── requirements.txt        <- The requirements file for reproducing the project
├── ResNet50_CIFAR10.ipynb  <- ResNet Training
├── VGG16_CIFAR10.ipynb     <- VGG16 Training
├── ViT_CIFAR10.ipynb       <- ViT Training
├── train_vit_kd.py         <- Script to run multiple ViT with KD
├── train_vit.py            <- Script to run multiple ViT no KD
├── train_deit_kd.py        <- Script to run multiple DeiT no KD
├── rtsuite.py              <- Test file for debugging and testing project
├── dis_results.py          <- Generate plots from Results
├── src                     <- Source code for use in this project
│   ├── config.py           <- training configurations
│   ├── data                <- Scripts to turn raw data into features for modeling
│   │   └── dataset.py
│   ├── models              <- Scripts to train models and then use trained models to make
│   │   ├── loss.py         <- Built Loss Functions
│   │   ├── model.py        <- Built Model
│   │   ├── test.py         <- Tests
│   │   └── train.py        <- Custom Training Pipelines
│   └── visualization       <- Scripts to create visualizations
│       └── visualize.py
└── LICENSE
```
