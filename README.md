# CPS803_Final_Project

## How to Run

Run VGG19_CIFAR10.ipynb to generate teacher model used

```bash
python main.py
```

## Installing Requirements:

Install PyTorch from: https://pytorch.org/get-started/locally/

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
```

```bash
pip install
```

```
├── README.md               <- README for developers using this project.
├── data                    <- Data used for this project
├── trained_models          <- Trained and serialized models
├── reports                 <- Generated analysis as PDF & LaTeX
├── requirements.txt        <- The requirements file for reproducing the project
├── src                     <- Source code for use in this project
│   ├── data                <- Scripts to turn raw data into features for modeling
│   │   └── dataset.py
│   ├── models              <- Scripts to train models and then use trained models to make
│   │   ├── loss.py
│   │   ├── model.py
│   │   ├── test.py
│   │   └── train.py
│   └── visualization       <- Scripts to create visualizations
│       └── visualize.py
└── LICENSE
```
