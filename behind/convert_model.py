import torch
from torch import nn
from torch.nn import functional as F

class ResidualBlock(nn.Module):  # @save
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, stride=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


class ResNet18(nn.Module):
    def __init__(self, num_class):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(ResidualBlock(64, 64, False),
                                    ResidualBlock(64, 64, False))

        self.layer2 = nn.Sequential(ResidualBlock(64, 128, True, 2),
                                    ResidualBlock(128, 128, False))

        self.layer3 = nn.Sequential(ResidualBlock(128, 256, True, 2),
                                    ResidualBlock(256, 256, False))

        self.layer4 = nn.Sequential(ResidualBlock(256, 512, True, 2),
                                    ResidualBlock(512, 512, False))

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Linear(512, num_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)  # 将多维矩阵展平为2维
        x = self.fc(x)

        return x


model = torch.load('../model/model.pth', map_location='cpu')

# 设置为评估模式
model.eval()

# 冻结所有层
for param in model.parameters():
    param.requires_grad = False

# 设置为评估模式
model.eval()

# 定义虚拟输入
x = torch.randn(1, 3, 224, 224)

# 导出模型
torch.onnx.export(model, x, '../model/model.onnx')
print('convert success！')