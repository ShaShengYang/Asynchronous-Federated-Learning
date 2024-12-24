from torchvision import datasets, transforms
from datasets.data_utils import read_client_data
import settings


def set_dataset(clients, dataset_name):
    if "cifar10" in dataset_name:
        for i in clients:
            i.train_dataset, i.test_dataset = read_client_data(dataset_name, i.client_id, settings.ALPHA_NONIID)
        print("Clients' train_datasets size: " + " ".join(str(len(_.train_dataset)) for _ in clients))
        print("Clients' test_datasets size: " + " ".join(str(len(_.test_dataset)) for _ in clients))
    elif 'mnist' in dataset_name:
        for i in clients:
            i.train_dataset, i.test_dataset = read_client_data(dataset_name, i.client_id, settings.ALPHA_NONIID)
        print("Clients' train_datasets size: " + " ".join(str(len(_.train_dataset)) for _ in clients))
        print("Clients' test_datasets size: " + " ".join(str(len(_.test_dataset)) for _ in clients))
    else:
        pass


def set_test_dataset():
    if "cifar10" in settings.DATASET:
        apply_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_dataset = datasets.CIFAR10("./datasets/dataset_cifar10", train=False, download=True,
                                        transform=apply_transform)
        return test_dataset
    elif "mnist" in settings.DATASET:
        apply_transform = transforms.ToTensor()
        test_dataset = datasets.MNIST("./datasets/dataset_mnist", train=False, download=True, transform=apply_transform)
        return test_dataset
