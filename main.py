import logging

import torch

import src.config as config
from src.data.dataset import get_dataloader
from src.models.model import DeiT, ViT, VGG16_classifier
from src.models.train import train_student
from src.models.loss import Hard_Distillation_Loss
from src.visualization.visualize import plot_sequential


torch.manual_seed(config.SEED)
torch.cuda.manual_seed(config.SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device being used: {}".format(DEVICE))


if __name__ == "__main__":

    # Get Data Loaders

    train_loader, val_loader, test_loader = get_dataloader("./data/CIFAR10/", config.BATCH_SIZE)

    print("Train Dataset Length: {}".format(len(train_loader)))
    print("Validation Dataset Length: {}".format(len(val_loader)))
    print("Test Dataset Length: {}".format(len(test_loader)))

    # Get Teacher Model (VGG19 Transfer Learning for CIFAR-10)

    classes = 10
    hidden_size = 512
    dropout = 0.3

    teacher_model = VGG16_classifier(classes, hidden_size, preprocess_flag=False, dropout=dropout)
    teacher_model.load_state_dict(torch.load("./trained_models/vgg16_cifar10.pt"))
    teacher_model.preprocess_flag = False

    # Model Hyper-parameters

    image_size = 32
    channel_size = 3
    patch_size = 4
    embed_size = 512
    num_heads = 8
    classes = 10
    num_layers = 3
    hidden_size = 512
    dropout = 0.2

    # Instantiate Model

    model = DeiT(image_size=image_size,
                 channel_size=channel_size,
                 patch_size=patch_size,
                 embed_size=embed_size,
                 num_heads=num_heads,
                 classes=classes,
                 num_layers=num_layers,
                 hidden_size=hidden_size,
                 teacher_model=teacher_model,
                 dropout=dropout
                 ).to(DEVICE)
    print(model)

    # Training

    criterion = Hard_Distillation_Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    loss_hist = train_student(model, train_loader, val_loader, criterion, optimizer, config, DEVICE)

    # Plot Train Stats

    plot_sequential(loss_hist["train accuracy"], "Epoch", "Train Accuracy")
    plot_sequential(loss_hist["train loss"], "Epoch", "Train Loss")
    plot_sequential(loss_hist["val accuracy"], "Epoch", "Validation Accuracy")

    # Save

    torch.save(model.state_dict(), './trained_models/DeiT.pt')

    print("Program has Ended")
