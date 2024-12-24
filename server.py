import queue
import settings
import model
import torch
import copy
import threading
from collections import OrderedDict


class Server:
    def __init__(self, update_times, printer):
        self.update_times = update_times
        self.data_queue = queue.Queue()  # 用户发给服务器的数据队列 FIFO
        self.running = True
        self.clients = None  # 用户列表 每个元素都是Client类的实例
        self.model = None
        self.weights = None
        self.device = torch.device("cuda:0")
        self.asynchronous_state = settings.ASYNCHRONOUS
        self.condition = threading.Condition()
        self.printer = printer

    def run(self, clients):
        epoch = 0
        self.clients = clients
        self.__pre_treat()  # 设置模型 获取模型初始权重 发送权重到每个用户
        if self.asynchronous_state:
            self.printer.output_queue.put("Server: Starting in asynchronous way")
            # 开始进行权重更新
            while epoch < self.update_times:
                # 从用户接收权重
                client, client_weights = self.data_queue.get()

                epoch += 1

                # 更新权重
                self.__update_weights_asynchronously(client_weights[0], epoch, client_weights[1])  # 0是权重 1是tau
                torch.save(self.weights, f"./logs/asynchronous/{epoch}.pth")
                self.printer.output_queue.put(f"Server: Data updated from {client.client_id}, epoch: {epoch}")
                # 将更新好的权重发给客户端 元组 权重 当前轮次
                self.__send_weights_to_client(client, epoch)

        else:
            self.printer.output_queue.put("Server: Starting in synchronous way")

            # 开始进行权重更新
            while epoch < self.update_times:
                with self.condition:
                    self.condition.wait_for(lambda: self.data_queue.qsize() == len(clients))

                # 列表每个元素是client_weight 每次get的是(client, (self.weights, self.server_time))
                weights_list = [self.data_queue.get()[1][0] for _ in range(len(clients))]

                epoch += 1

                self.__update_weights_synchronously(weights_list)
                torch.save(self.weights, f"./logs/synchronous/{epoch}.pth")
                self.printer.output_queue.put(f"Server: Data updated, epoch: {epoch}")
                # 给每个用户发更新后的权重
                for client in self.clients:
                    self.__send_weights_to_client(client, epoch)

        # 停止并且给用户发停止信号
        for client in self.clients:
            client.response_queue.put("STOP")

        self.printer.output_queue.put("Server: Stopping")
        self.running = False

    def receive_data(self, client, data):
        self.data_queue.put((client, data))

    def __setup_model(self):
        if settings.MODEL == "vgg_cifar":
            self.model = model.VggCifar()
            self.model.to(self.device)
        elif settings.MODEL == "resnet_cifar":
            self.model = model.ResNet18()
            self.model.to(self.device)
        elif settings.MODEL == "vgg_mnist":
            self.model = model.VggMnist()
            self.model.to(self.device)
        else:
            pass

    def __update_weights_asynchronously(self, client_weights, epoch, tau):
        alpha = 0.8
        a = pow((epoch - tau + 1), -0.5)
        alpha_t = alpha * a
        new_weights = OrderedDict()
        for key in self.weights:
            if "num_batches_tracked" in key:
                new_weights[key] = client_weights[key]
            else:
                new_weights[key] = (1 - alpha_t) * self.weights[key] + alpha_t * client_weights[key]
        self.weights = new_weights

    def __update_weights_synchronously(self, weights_list):
        new_weights = copy.deepcopy(weights_list[0])
        for key in new_weights:
            if "num_batches_tracked" in key:
                new_weights[key] = weights_list[0][key]
            else:
                for i in range(1, len(weights_list)):
                    new_weights[key] += weights_list[i][key]
                new_weights[key] /= len(weights_list)
        self.weights = new_weights

    def __send_weights_to_client(self, client, epoch):
        # 权重deepcopy一下再发送 避免发生问题
        client.response_queue.put((copy.deepcopy(self.weights), epoch))

    def __pre_treat(self):
        self.__setup_model()
        self.weights = self.model.get_weights()  # 设置初始权重
        for client in self.clients:
            self.__send_weights_to_client(client, 0)


class Printer:
    def __init__(self):
        self.output_queue = queue.Queue()

    def run(self):
        print("#" * 50)
        while True:
            info = self.output_queue.get()
            if info == "STOP":
                break
            else:
                print(info)
