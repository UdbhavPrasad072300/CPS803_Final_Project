from torch.utils.data import random_split, DataLoader

from torchvision import datasets, transforms


def getTransform():
    mean, std = (0.5,), (0.5,)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
    )

    return transform


def get_dataset(directory, transform):
    train_set = datasets.CIFAR10(directory, download=True, train=True, transform=transform)
    test_set = datasets.CIFAR10(directory, download=True, train=False, transform=transform)

    length = int(len(test_set) / 2)

    val_set, test_set = random_split(test_set, [length, length])

    return train_set, val_set, test_set


def make_dataloader(train_set, val_set, test_set, batch_size):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def get_dataloader(directory, batch_size):
    return make_dataloader(*get_dataset(directory, getTransform()), batch_size=batch_size)
