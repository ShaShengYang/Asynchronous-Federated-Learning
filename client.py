import copy
import queue
import settings
import model
import torch
from torch.utils.data import DataLoader


class Client:
    def __init__(self, server, client_id, printer):
        self.server = server
        self.client_id = client_id
        self.model = None
        self.loss_function = None
        self.optimizer = None
        self.weights = None
        self.learning_rate = settings.LEARNINGRATE
        self.response_queue = queue.Queue()  # 接收服务器信息的队列
        self.device = torch.device("cuda:0")
        self.train_dataset = None
        self.test_dataset = None
        self.batch_size = settings.BATCH_SIZE
        self.server_time = 0  # 记录接受参数时服务器的更新轮数
        self.printer = printer

    def run(self):
        self.__pre_treat()  # 设置模型 获取初始参数 设置损失函数与优化器 设置数据集
        self.printer.output_queue.put(f"Client {self.client_id}: Starting")

        while True:
            self.__set_weights(self.weights)
            self.__update_weights()  # 训练并更新权重
            self.printer.output_queue.put(f"Client {self.client_id}: Updated weights and sent to Server")
            # 向服务器发送权重 上一次接收权重时服务器的轮次 deepcopy一下防止出现问题
            self.server.receive_data(self, (copy.deepcopy(self.weights), self.server_time))
            with self.server.condition:
                if self.server.data_queue.qsize() == len(self.server.clients):
                    self.server.condition.notify()
                else:
                    pass
            response = self.response_queue.get()  # 从服务器接收权重
            if response == "STOP":
                break
            else:
                self.weights = response[0]
                self.server_time = response[1]
            self.printer.output_queue.put(f"Client {self.client_id}: got new weights from Server")

        self.printer.output_queue.put(f"Client {self.client_id}: Stopping")

    def __pre_treat(self):
        self.__set_model()
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.loss_function.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        response = self.response_queue.get()
        self.weights = response[0]  # 收到的是经过deepcopy后的 随便用 不会出问题
        self.server_time = response[1]

    def __set_model(self):
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

    def __set_weights(self, weights):
        self.model.load_state_dict(weights)

    def __update_weights(self):
        train_dataloader = DataLoader(self.train_dataset, self.batch_size)
        test_dataloader = DataLoader(self.test_dataset, self.batch_size)
        for features, labels in train_dataloader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model.forward(features)
            loss = self.loss_function(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.weights = self.model.state_dict()

        test_loss = 0
        true_count = 0
        with torch.no_grad():
            for features, labels in test_dataloader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model.forward(features)
                loss = self.loss_function(outputs, labels)
                test_loss += loss.item()
                temp_list = outputs.argmax(1) == labels
                temp_true_count = temp_list.sum().item()
                true_count += temp_true_count
        test_accuracy = true_count / len(self.test_dataset) * 100

        self.printer.output_queue.put(
            f"Client {self.client_id} has finished train, accuracy is {test_accuracy}%, total loss is {test_loss}"
        )
