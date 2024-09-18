import torch
import torch.nn as nn
import torch.optim as optim
from torch.ao.quantization import get_default_qat_qconfig, prepare_qat, convert
from torchvision import datasets, transforms
from torch.quantization import QuantStub, DeQuantStub, fuse_modules


# 定义一个简单的卷积神经网络
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
        self.fc2 = nn.Linear(128, 10)
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


def prepare_data_loaders():
    # 设置数据加载器
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=64, shuffle=True)
    return train_loader


def prepare_test_loader():
    # 设置测试数据加载器
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=64, shuffle=False)
    return test_loader


# QAT 模型训练函数
# def train(model, device, train_loader, optimizer, criterion, epoch):
#     print('------------training------------')
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % 100 == 0:
#             print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
#                   f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    print('------------training------------')
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        total_batches = len(train_loader)

        for inputs, labels in train_loader:
            # Move data to the appropriate device (GPU or CPU)
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.to(device)
            # Zero the gradients before each batch
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate the loss
            running_loss += loss.item()

        # Print average loss for the epoch
        avg_loss = running_loss / total_batches
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')


# 评估函数
def evaluate(model, device, test_loader):
    print('------------evaluating------------')
    model.eval()  # 设置为评估模式
    model.to(device)
    correct = 0
    total = 0

    with torch.no_grad():  # 在评估时禁用梯度计算
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)

            # 获取预测结果
            _, predicted = torch.max(outputs.data, 1)

            # 更新总数和正确预测的数量
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 计算准确率
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    return accuracy


def main():
    # 使用 GPU
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu:0")
    train_loader = prepare_data_loaders()
    test_loader = prepare_test_loader()
    # 训练模型
    num_epochs = 2
    # 创建模型实例并移动到 GPU
    model = SimpleCNN().to(cuda_device)
    # 在进行 QAT 前，首先进行模型层融合
    model.fuse_model()
    # 为 QAT 设置 QConfig，指定使用 per_tensor_affine 量化方案
    model.qconfig = torch.ao.quantization.QConfig(
        activation=torch.ao.quantization.FakeQuantize.with_args(observer=torch.ao.quantization.MinMaxObserver,
                                                                qscheme=torch.per_tensor_affine),
        weight=torch.ao.quantization.default_weight_fake_quant
    )
    # model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    # 准备 QAT
    model = prepare_qat(model, inplace=True)

    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    # 训练
    # for epoch in range(1, num_epochs + 1):
    #     train(model, device, train_loader, optimizer, criterion, epoch)
    train_model(model, train_loader, criterion, optimizer, num_epochs, cuda_device)

    # 一定要在convert之前将模型移动到cpu上，不然无法evaluate
    model.to(cpu_device)
    # 将 QAT 模型转换为量化模型
    model = convert(model.eval(), inplace=True)
    # model.to(cpu_device)  # 将模型移动到CPU

    # 评估时使用CPU
    # 评估量化后的模型
    evaluate(model, cpu_device, test_loader)
    # 保存量化模型
    torch.save(model.state_dict(), 'quantized_model.pth')

    print("QAT 量化完成，模型已保存。")


if __name__ == "__main__":
    main()
