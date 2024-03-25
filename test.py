import threading
from torchvision import transforms, datasets
import torch
from torch.utils import data
from server import Server, Printer
from client import Client
from datasets import process_dataset
import model
import settings


def evaluate_accuracy():
    device = torch.device("cuda:0")
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn.to(device)
    apply_transform = transforms.ToTensor()
    if settings.DATASET == "cifar":
        test_dataset = datasets.CIFAR10("./datasets/dataset_cifar10", train=False, download=True,
                                        transform=apply_transform)
    elif settings.DATASET == 'mnist':
        test_dataset = datasets.MNIST("./datasets/dataset_mnist", train=False, transform=apply_transform,
                                      download=True, )

    test_dataloader = data.DataLoader(test_dataset, 64)

    if settings.MODEL == "resnet_cifar":
        test_model = model.ResNet18()
    elif settings.MODEL == "vgg_cifar":
        test_model = model.VggCifar()
    elif settings.MODEL == "vgg_mnist":
        test_model = model.VggMnist()

    test_model.to(device)
    if settings.ASYNCHRONOUS == False:
        test_model.load_state_dict(torch.load(f"./logs/synchronous/{settings.UPDATE_TIMES}.pth"))
    else:
        test_model.load_state_dict(torch.load(f"./logs/asynchronous/{settings.UPDATE_TIMES}.pth"))

    test_loss = 0
    true_count = 0
    with torch.no_grad():
        for features, labels in test_dataloader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = test_model.forward(features)
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()
            temp_list = outputs.argmax(1) == labels
            temp_true_count = temp_list.sum().item()
            true_count += temp_true_count
            test_accuracy = true_count / len(test_dataset) * 100
    print(f"After {settings.UPDATE_TIMES} times aggregation, model's accuracy is {test_accuracy}%")

def main():
    clients = []  # 用户列表 每个元素为Client类的实例
    client_threads = []  # 用户线程列表 用于启动线程
    printer = Printer()
    server = Server(settings.UPDATE_TIMES, printer)
    for i in range(settings.NUM_CLIENTS):
        client = Client(server, i, printer)
        clients.append(client)
        client_threads.append(threading.Thread(target=client.run))
    process_dataset.set_dataset(clients, dataset_name=settings.DATASET)

    printer_thread = threading.Thread(target=printer.run)
    printer_thread.start()
    server_thread = threading.Thread(target=server.run, args=(clients,))
    server_thread.start()
    for i in client_threads:
        i.start()

    server_thread.join()
    for thread in client_threads:
        thread.join()
    printer.output_queue.put("STOP")
    printer_thread.join()


if __name__ == "__main__":
    main()
