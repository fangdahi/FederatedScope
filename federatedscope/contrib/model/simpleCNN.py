import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
from federatedscope.register import register_model


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.quant = QuantStub()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(9216, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 62)
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2)(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = self.dequant(x)
        return x

    # 模型层融合
    def fuse_model(self):
        # 这里将 Conv 和 ReLU 层进行融合
        fuse_modules(self, [['conv1', 'relu1'], ['conv2', 'relu2']], inplace=True)


def call_simple_cnn(model_config, local_data):
    if model_config.type == "SimpleCNN":
        model = SimpleCNN()
        return model


register_model("SimpleCNN", call_simple_cnn)
