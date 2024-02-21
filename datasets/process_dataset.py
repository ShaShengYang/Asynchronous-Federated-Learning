import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random


def set_dataset(clients, dataset_name):
    if dataset_name == "cifar":
        apply_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR10("./datasets/dataset_cifar10", train=True, download=True,
                                         transform=apply_transform)
        test_dataset = datasets.CIFAR10("./datasets/dataset_cifar10", train=False, download=True,
                                        transform=apply_transform)
        # 计算出平均下来每个用户的数据集大小
        train_dataset_size = len(train_dataset)
        test_dataset_size = len(test_dataset)
        sub_train_dataset_size = train_dataset_size // len(clients)  # 除不尽的取整 剩点不要了
        sub_test_dataset_size = test_dataset_size // len(clients)

        train_dataset_indices = torch.randperm(train_dataset_size).tolist()
        test_dataset_indices = torch.randperm(test_dataset_size).tolist()
        sub_train_datasets = [
            torch.utils.data.Subset(train_dataset, train_dataset_indices[i: i + sub_train_dataset_size])
            for i in range(0, sub_train_dataset_size * len(clients), sub_train_dataset_size)]

        sub_test_datasets = [
            torch.utils.data.Subset(test_dataset, test_dataset_indices[i: i + sub_test_dataset_size])
            for i in range(0, sub_test_dataset_size * len(clients), sub_test_dataset_size)]

        for i, j in zip(clients, sub_train_datasets):
            i.train_dataset = j
        for i, j in zip(clients, sub_test_datasets):
            i.test_dataset = j

        print("Clients' train_datasets size: " + " ".join(str(len(i)) for i in sub_train_datasets))
        print("Clients' test_datasets size: " + " ".join(str(len(i)) for i in sub_test_datasets))

    elif dataset_name == "mnist":
        apply_transform = transforms.ToTensor()
        train_dataset = datasets.MNIST("./datasets/dataset_mnist", train=True, download=True, transform=apply_transform)
        test_dataset = datasets.MNIST("./datasets/dataset_mnist", train=False, download=True, transform=apply_transform)
        # 计算出平均下来每个用户的数据集大小
        train_dataset_size = len(train_dataset)
        test_dataset_size = len(test_dataset)
        sub_train_dataset_size = train_dataset_size // len(clients)  # 除不尽的取整 剩点不要了
        sub_test_dataset_size = test_dataset_size // len(clients)

        train_dataset_indices = torch.randperm(train_dataset_size).tolist()
        test_dataset_indices = torch.randperm(test_dataset_size).tolist()
        sub_train_datasets = [
            torch.utils.data.Subset(train_dataset, train_dataset_indices[i: i + sub_train_dataset_size])
            for i in range(0, sub_train_dataset_size * len(clients), sub_train_dataset_size)]

        sub_test_datasets = [
            torch.utils.data.Subset(test_dataset, test_dataset_indices[i: i + sub_test_dataset_size])
            for i in range(0, sub_test_dataset_size * len(clients), sub_test_dataset_size)]

        for i, j in zip(clients, sub_train_datasets):
            i.train_dataset = j
        for i, j in zip(clients, sub_test_datasets):
            i.test_dataset = j
        print("Clients' train_datasets size: " + " ".join(str(len(i)) for i in sub_train_datasets))
        print("Clients' test_datasets size: " + " ".join(str(len(i)) for i in sub_test_datasets))

    elif dataset_name == "mnist_noniid":
        apply_transform = transforms.ToTensor()
        train_dataset = datasets.MNIST("./datasets/dataset_mnist", train=True, download=True,
                                       transform=apply_transform)
        test_dataset = datasets.MNIST("./datasets/dataset_mnist", train=False, download=True,
                                      transform=apply_transform)
        # 计算出平均下来每个用户的数据集大小
        train_dataset_size = len(train_dataset)
        test_dataset_size = len(test_dataset)
        sub_train_dataset_size = train_dataset_size // len(clients)  # 除不尽的取整 剩点不要了
        sub_test_dataset_size = test_dataset_size // len(clients)

        train_dataset_indices = torch.randperm(train_dataset_size).tolist()
        test_dataset_indices = torch.randperm(test_dataset_size).tolist()

        sub_train_datasets = [
            train_dataset_indices[i: i + sub_train_dataset_size]
            for i in range(0, sub_train_dataset_size * len(clients), sub_train_dataset_size)]

        sub_test_datasets = [
            torch.utils.data.Subset(test_dataset, test_dataset_indices[i: i + sub_test_dataset_size])
            for i in range(0, sub_test_dataset_size * len(clients), sub_test_dataset_size)]

        for idx, indices in enumerate(sub_train_datasets):
            exclude_labels = set(random.sample(range(10), 3))  # 随机选3个标签作为要排除的标签
            # 选出不包含在exclude_labels中的indices
            filtered_indices = [x for x in indices if train_dataset[x][1] not in exclude_labels]
            # 用过滤后的indices创建Subset
            sub_train_datasets[idx] = torch.utils.data.Subset(train_dataset, filtered_indices)

        for i, j in zip(clients, sub_train_datasets):
            i.train_dataset = j
        for i, j in zip(clients, sub_test_datasets):
            i.test_dataset = j
        print("Clients' train_datasets size: " + " ".join(str(len(i)) for i in sub_train_datasets))
        print("Clients' test_datasets size: " + " ".join(str(len(i)) for i in sub_test_datasets))


if __name__ == "__main__":
    # transform = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # # train_dataset是一个个元组 第一个元素是图片信息3 32 32 第二个是标签
    # dataset = datasets.CIFAR10('./dataset_cifar10', train=True, download=True, transform=transform)
    # # train_loader是一个个列表 第一个元素是32 3 32 32 第二个是标签
    # train_loader = DataLoader(dataset, 32)
    # count = 0
    # for x in dataset:
    #     if count < 5:
    #         print(x)
    #     else:
    #         break
    #     count += 1
    print("helki")
