import numpy as np
import os
import torch
import torchvision
import torchvision.transforms as transforms
from datasets.dataset_utils import separate_data, split_data, save_file
from torch.utils.data import DataLoader
import settings


def generate_cifar10(dir_path, num_clients, num_classes, alpha):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    # Setup directory for train/test data
    train_path = "./cifar10_noniid/" + f"{alpha}/"
    test_path = "./cifar10_noniid/" + f"{alpha}/"
    if not os.path.exists(train_path):
        os.makedirs(train_path)

    transform_train = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(
        root=dir_path, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root=dir_path, train=False, download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)
    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, alpha)
    train_data, test_data = split_data(X, y)
    save_file(train_path, test_path, train_data, test_data)


def generate_mnist(dir_path, num_clients, num_classes, alpha):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    train_path = "./mnist_noniid/" + f"{alpha}/"
    test_path = "./mnist_noniid/" + f"{alpha}/"

    if not os.path.exists(train_path):
        os.makedirs(train_path)

    transform_train = transforms.ToTensor()
    transform_test = transforms.ToTensor()

    trainset = torchvision.datasets.MNIST(
        root=dir_path, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.MNIST(
        root=dir_path, train=False, download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)
    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, alpha)
    train_data, test_data = split_data(X, y)
    save_file(train_path, test_path, train_data, test_data)


if __name__ == "__main__":
    if settings.DATASET == 'cifar10_noniid':
        generate_cifar10("./dataset_cifar10/", settings.NUM_CLIENTS, 10, settings.ALPHA_NONIID)
        print("Non-iid alpha: ", settings.ALPHA_NONIID)
    elif settings.DATASET == 'cifar10':
        generate_cifar10("./dataset_cifar10/", settings.NUM_CLIENTS, 10, 1000)
        print("Non-iid alpha: 1000")
    elif settings.DATASET == 'mnist_noniid':
        generate_mnist("./dataset_mnist/", settings.NUM_CLIENTS, 10, settings.ALPHA_NONIID)
        print("Non-iid alpha: ", settings.ALPHA_NONIID)
    elif settings.DATASET == 'mnist':
        generate_mnist("./dataset_mnist/", settings.NUM_CLIENTS, 10, 1000)
        print("Non-iid alpha: 1000")
    else:
        pass
