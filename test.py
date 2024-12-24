import threading
import numpy as np
import torch
from torch.utils import data
from server import Server, Printer
from client import Client
from datasets import process_dataset
import model
import settings


def evaluate_accuracy(epoch, accuracy_list, loss_list, test_dataset):
    device = torch.device("cuda:0")
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn.to(device)
    test_dataloader = data.DataLoader(test_dataset, settings.BATCH_SIZE, drop_last=True)
    if settings.MODEL == 'resnet_cifar':
        test_model = model.ResNet18()
    elif settings.MODEL == 'vgg_cifar':
        test_model = model.VggCifar()
    else:
        test_model = model.VggMnist()
    test_model.to(device)
    if settings.ASYNCHRONOUS:
        test_model.load_state_dict(torch.load(f"./logs/asynchronous/{epoch}.pth"))
    else:
        test_model.load_state_dict(torch.load(f"./logs/synchronous/{epoch}.pth"))
    test_loss = 0
    correct_count = 0
    total_samples = 0
    with torch.no_grad():
        for features, labels in test_dataloader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = test_model.forward(features)
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_count += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    test_accuracy = correct_count / total_samples
    if epoch == settings.UPDATE_TIMES:
        print(f"After {epoch} times aggregation, model's accuracy is {test_accuracy}")
    accuracy_list.append(test_accuracy)
    loss_list.append(test_loss)


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
    experiment_name = "asyn"
    test_data = process_dataset.set_test_dataset()
    accuracies = []
    losses = []
    if settings.ASYNCHRONOUS:
        for j in range(settings.NUM_CLIENTS, settings.UPDATE_TIMES + 1, settings.NUM_CLIENTS):
            evaluate_accuracy(j, accuracies, losses, test_data)
            print(j)
    else:
        for j in range(1, settings.UPDATE_TIMES + 1):
            evaluate_accuracy(j, accuracies, losses, test_data)
    np.save(f'./logs/acc/accuracy_array_{experiment_name}_{settings.ALPHA_NONIID}.npy', accuracies)
    np.save(f'./logs/loss/loss_array_{experiment_name}_{settings.ALPHA_NONIID}.npy', losses)
