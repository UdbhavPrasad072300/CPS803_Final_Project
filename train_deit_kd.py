import logging

import torch
import torch.nn as nn

import src.config as config
from src.data.dataset import get_dataloader
from src.models.model import ViT, ResNet_classifier, DeiT, VGG16_classifier
from src.models.train import train_student
from src.models.loss import Hard_Distillation_Loss, Soft_Distillation_Loss
from src.visualization.visualize import plot_sequential
from src.models.test import test


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

    # Records

    test_acc = []

    # Get 5 ViT Models

    img_size = 32
    c_size = 3
    p_size = 8
    e_size = 512
    n_heads = 8
    classes = 10
    hidden_size = 256
    dropout = 0.3

    # criterion = Soft_Distillation_Loss(0.2, 2)
    criterion = Hard_Distillation_Loss()

    # Teacher

    # teacher_model = ResNet_classifier(10, 512, preprocess_flag=False, dropout=0.3)
    # teacher_model.load_state_dict(torch.load("./trained_models/resnet50_cifar10.pt"))
    # teacher_model.preprocess_flag = False

    teacher_model = VGG16_classifier(10, 512, preprocess_flag=False, dropout=0.3).to(DEVICE)
    teacher_model.load_state_dict(torch.load("./trained_models/vgg16_cifar10.pt"))
    teacher_model.preprocess_flag = False

    # 1 Encoder

    print("-" * 20, "ENCODER 1", "-" * 20)
    model = DeiT(img_size, c_size, p_size, e_size, n_heads, classes, 1, hidden_size, dropout=dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    loss_hist_1 = train_student(model, train_loader, val_loader, criterion, optimizer, config, teacher_model, DEVICE)
    test_acc.append(test(model, test_loader, DEVICE))
    torch.save(model.state_dict(), './trained_models/vit_encoder_resnet_kd_1.pt')

    # 2 Encoder

    print("-" * 20, "ENCODER 2", "-" * 20)
    model = DeiT(img_size, c_size, p_size, e_size, n_heads, classes, 2, hidden_size, dropout=dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    loss_hist_2 = train_student(model, train_loader, val_loader, criterion, optimizer, config, teacher_model, DEVICE)
    test_acc.append(test(model, test_loader, DEVICE))
    torch.save(model.state_dict(), './trained_models/vit_encoder_resnet_kd_2.pt')

    # 3 Encoder

    print("-" * 20, "ENCODER 3", "-" * 20)
    model = DeiT(img_size, c_size, p_size, e_size, n_heads, classes, 3, hidden_size, dropout=dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    loss_hist_3 = train_student(model, train_loader, val_loader, criterion, optimizer, config, teacher_model, DEVICE)
    test_acc.append(test(model, test_loader, DEVICE))
    torch.save(model.state_dict(), './trained_models/vit_encoder_resnet_kd_3.pt')

    # 4 Encoder

    print("-" * 20, "ENCODER 4", "-" * 20)
    model = DeiT(img_size, c_size, p_size, e_size, n_heads, classes, 4, hidden_size, dropout=dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    loss_hist_4 = train_student(model, train_loader, val_loader, criterion, optimizer, config, teacher_model, DEVICE)
    test_acc.append(test(model, test_loader, DEVICE))
    torch.save(model.state_dict(), './trained_models/vit_encoder_resnet_kd_4.pt')

    # Plot Train Stats

    plot_sequential(loss_hist_1["train accuracy"], "1 Encoder - DeiT - Resnet KD", "Epoch", "Train Accuracy")
    plot_sequential(loss_hist_1["train loss"], "1 Encoder - DeiT - Resnet KD", "Epoch", "Train Loss")
    plot_sequential(loss_hist_1["val accuracy"], "1 Encoder - DeiT - Resnet KD", "Epoch", "Validation Accuracy")

    plot_sequential(loss_hist_2["train accuracy"], "2 Encoder - DeiT - Resnet KD", "Epoch", "Train Accuracy")
    plot_sequential(loss_hist_2["train loss"], "2 Encoder - DeiT - Resnet KD", "Epoch", "Train Loss")
    plot_sequential(loss_hist_2["val accuracy"], "2 Encoder - DeiT - Resnet KD", "Epoch", "Validation Accuracy")

    plot_sequential(loss_hist_3["train accuracy"], "3 Encoder - DeiT - Resnet KD", "Epoch", "Train Accuracy")
    plot_sequential(loss_hist_3["train loss"], "3 Encoder - DeiT - Resnet KD", "Epoch", "Train Loss")
    plot_sequential(loss_hist_3["val accuracy"], "3 Encoder - DeiT - Resnet KD", "Epoch", "Validation Accuracy")

    plot_sequential(loss_hist_4["train accuracy"], "4 Encoder - DeiT", "Epoch", "Train Accuracy")
    plot_sequential(loss_hist_4["train loss"], "4 Encoder - DeiT", "Epoch", "Train Loss")
    plot_sequential(loss_hist_4["val accuracy"], "4 Encoder - DeiT", "Epoch", "Validation Accuracy")

    plot_sequential(test_acc, "Test Accuracies - Encoder 1-5 - DeiT - Resnet KD", "Encoder Num", "Test Accuracy")

    # ResNet - Hard Label = [52.34, 57.78, 55.64, 57.66]
    # ResNet - Soft Label = [38.1, 51.66, 55.22, 55.04]
    # VGG - Hard Label = [52.26, 57.76, 58.12, 23.46]
    # VGG - Soft Label = [45.26, 54.02, 56.52, 54.36]
    print(test_acc)

    print("Program has Ended")
