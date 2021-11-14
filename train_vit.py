import logging

import torch
import torch.nn as nn

import src.config as config
from src.data.dataset import get_dataloader
from src.models.model import ViT
from src.models.train import train
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
    p_size = 16
    e_size = 256
    n_heads = 8
    classes = 10
    hidden_size = 256
    dropout = 0.3

    criterion = nn.CrossEntropyLoss()

    # 1 Encoder

    print("-" * 20, "ENCODER 1", "-" * 20)
    model = ViT(img_size, c_size, p_size, e_size, n_heads, classes, 1, hidden_size, dropout=dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    loss_hist_1 = train(model, train_loader, val_loader, criterion, optimizer, config, DEVICE)
    test_acc.append(test(model, test_loader, DEVICE))
    torch.save(model.state_dict(), './trained_models/vit_encoder_1.pt')

    plot_sequential(loss_hist_1["train accuracy"], "1 Encoder - ViT", "Epoch", "Train Accuracy")
    plot_sequential(loss_hist_1["train loss"], "1 Encoder - ViT", "Epoch", "Train Loss")
    plot_sequential(loss_hist_1["val accuracy"], "1 Encoder - ViT", "Epoch", "Validation Accuracy")

    # 2 Encoder

    print("-" * 20, "ENCODER 2", "-" * 20)
    model = ViT(img_size, c_size, p_size, e_size, n_heads, classes, 2, hidden_size, dropout=dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    loss_hist_2 = train(model, train_loader, val_loader, criterion, optimizer, config, DEVICE)
    test_acc.append(test(model, test_loader, DEVICE))
    torch.save(model.state_dict(), './trained_models/vit_encoder_2.pt')

    # 3 Encoder

    print("-" * 20, "ENCODER 3", "-" * 20)
    model = ViT(img_size, c_size, p_size, e_size, n_heads, classes, 3, hidden_size, dropout=dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    loss_hist_3 = train(model, train_loader, val_loader, criterion, optimizer, config, DEVICE)
    test_acc.append(test(model, test_loader, DEVICE))
    torch.save(model.state_dict(), './trained_models/vit_encoder_3.pt')

    # 4 Encoder

    print("-" * 20, "ENCODER 4", "-" * 20)
    model = ViT(img_size, c_size, p_size, e_size, n_heads, classes, 4, hidden_size, dropout=dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    loss_hist_4 = train(model, train_loader, val_loader, criterion, optimizer, config, DEVICE)
    test_acc.append(test(model, test_loader, DEVICE))
    torch.save(model.state_dict(), './trained_models/vit_encoder_4.pt')

    # 5 Encoder

    print("-" * 20, "ENCODER 5", "-" * 20)
    model = ViT(img_size, c_size, p_size, e_size, n_heads, classes, 5, hidden_size, dropout=dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    loss_hist_5 = train(model, train_loader, val_loader, criterion, optimizer, config, DEVICE)
    test_acc.append(test(model, test_loader, DEVICE))
    torch.save(model.state_dict(), './trained_models/vit_encoder_5.pt')

    # Plot Train Stats

    plot_sequential(loss_hist_2["train accuracy"], "2 Encoder - ViT", "Epoch", "Train Accuracy")
    plot_sequential(loss_hist_2["train loss"], "2 Encoder - ViT", "Epoch", "Train Loss")
    plot_sequential(loss_hist_2["val accuracy"], "2 Encoder - ViT", "Epoch", "Validation Accuracy")

    plot_sequential(loss_hist_3["train accuracy"], "3 Encoder - ViT", "Epoch", "Train Accuracy")
    plot_sequential(loss_hist_3["train loss"], "3 Encoder - ViT", "Epoch", "Train Loss")
    plot_sequential(loss_hist_3["val accuracy"], "3 Encoder - ViT", "Epoch", "Validation Accuracy")

    plot_sequential(loss_hist_4["train accuracy"], "4 Encoder - ViT", "Epoch", "Train Accuracy")
    plot_sequential(loss_hist_4["train loss"], "4 Encoder - ViT", "Epoch", "Train Loss")
    plot_sequential(loss_hist_4["val accuracy"], "4 Encoder - ViT", "Epoch", "Validation Accuracy")

    plot_sequential(loss_hist_5["train accuracy"], "5 Encoder - ViT", "Epoch", "Train Accuracy")
    plot_sequential(loss_hist_5["train loss"], "5 Encoder - ViT", "Epoch", "Train Loss")
    plot_sequential(loss_hist_5["val accuracy"], "5 Encoder - ViT", "Epoch", "Validation Accuracy")

    plot_sequential(test_acc, "Test Accuracies - Encoder 1-5 - ViT", "Encoder Num", "Test Accuracy")

    # [50.4, 55.64, 58.1, 58.12, 57.92]
    print(test_acc)

    print("Program has Ended")
