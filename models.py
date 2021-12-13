import torch.nn.functional as F
from torch.nn import Module, Conv2d, Linear, ReLU, MaxPool2d, Sequential, Dropout, ModuleList
import torch
import torchvision

CLASSES = 20


class Baseline(Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=10, kernel_size=(5, 5))
        self.conv2 = Conv2d(in_channels=10, out_channels=20, kernel_size=(5, 5))
        self.conv3 = Conv2d(in_channels=20, out_channels=20, kernel_size=(5, 5))
        self.conv4 = Conv2d(in_channels=20, out_channels=20, kernel_size=(5, 5))
        self.fc1 = Linear(in_features=320, out_features=50)
        self.fc2 = Linear(50, CLASSES)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class AlexNetBackBone(Module):
    def __init__(self):
        super().__init__()
        self.features = Sequential(
            Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=2),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2),
            Conv2d(64, 192, kernel_size=(5, 5), padding=2),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2),
            Conv2d(192, 384, kernel_size=(3, 3), padding=1),
            ReLU(inplace=True),
            Conv2d(384, 256, kernel_size=(3, 3), padding=1),
            ReLU(inplace=True),
            Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=2),
        )
        alex_net_pretrained = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
        self.features.load_state_dict(alex_net_pretrained.features.state_dict())

    def forward(self, x):
        x = self.features(x)
        return x

    @staticmethod
    def flattened_output_size(input_dim):
        if input_dim == 64:
            return 256 * 1 * 1
        elif input_dim == 128:
            return 256 * 3 * 3
        elif input_dim == 256:
            return 256 * 7 * 7


class VGG16(Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        vgg16_full = torchvision.models.vgg16(pretrained=True).eval()
        self.features = vgg16_full.features
        self.avgpool = vgg16_full.avgpool
        self.fc = vgg16_full.classifier[:5]
        self.fc_end = Linear(in_features=4096, out_features=20)
        self.dropout = Dropout(p=dropout)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(self.fc(x))
        x = self.fc_end(F.relu(x))
        return x


class RegNet(Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        regnet_full = torchvision.models.regnet_y_8gf(pretrained=True).eval()
        self.stem = regnet_full.stem
        self.trunk_output = regnet_full.trunk_output
        self.avgpool = regnet_full.avgpool
        self.fc = Sequential(Dropout(p=dropout),
                             Linear(in_features=2016, out_features=1024),
                             ReLU(),
                             Dropout(p=dropout),
                             Linear(in_features=1024, out_features=20))

    def forward(self, x):
        x = self.stem(x)
        x = self.trunk_output(x)
        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x


class ResNet(Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        resnet_full = torchvision.models.resnext50_32x4d(pretrained=True).eval()
        #resnet_full = torchvision.models.resnet152(pretrained=True).eval()
        self.conv1 = resnet_full.conv1
        self.bn1 = resnet_full.bn1
        self.relu = resnet_full.relu
        self.maxpool = resnet_full.maxpool
        self.layer1 = resnet_full.layer1
        self.layer2 = resnet_full.layer2
        self.layer3 = resnet_full.layer3
        self.layer4 = resnet_full.layer4
        self.avgpool = resnet_full.avgpool
        self.fc = Sequential(Dropout(p=dropout),
                             Linear(in_features=2048, out_features=20))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x


class EffNet(Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        effnet_full = torchvision.models.efficientnet_b4(pretrained=True).eval()
        self.features = effnet_full.features
        self.avgpool = effnet_full.avgpool
        self.fc = Sequential(Dropout(p=dropout),
                             Linear(in_features=1792, out_features=20))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x


class LinearClassifier(Module):
    def __init__(self, in_features, dropout=0.5):
        super(LinearClassifier, self).__init__()
        self.dropout = Dropout(p=dropout)
        self.fc = Linear(in_features=in_features, out_features=CLASSES)

    def forward(self, x):
        x = self.dropout(x)
        x = F.relu(self.fc(x))
        return x


class SumModel(Module):
    def __init__(self, model_paths, torchModelClasses):
        super(SumModel, self).__init__()
        checkpoints = [torch.load(path) for path in model_paths]
        self.models = ModuleList([torchModelClass() for torchModelClass in torchModelClasses])
        for model, checkpoint in list(zip(self.models, checkpoints)):
            model.load_state_dict(checkpoint)

    def forward(self, x):
        return torch.stack([m(x) for m in self.models]).sum(dim=0)
