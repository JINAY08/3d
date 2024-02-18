import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
# from .bfp_ops import BFPLinear, BFPConv2d, unpack_bfp_args
# from .bfp_optim import get_bfp_optim
import torch.optim as optim
from tqdm import tqdm, trange
import av
import math
from functools import partial
from torchvision.datasets.video_utils import VideoClips
# from subsetpreprocess import FrameGenerator, subset_paths
from pathlib import Path
from ucf101_subset import UCF101Dataset

PATH = './ucf101_net.pth'

num_format = 'posit'

class CustomConv3d(torch.nn.Conv3d):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride,padding, dilation, groups, bias)
        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.num_format = num_format
    def conditions_l1(self, xx):
        conditions, values = [], []
        x = torch.where(xx < 0, -1*xx, xx)
        for e in range(16):
            for j in range(8):
                if(j==0): 
                    conditions.append(torch.logical_and(x > 0, x < (1.0625/pow(2,16))))
                    mid = 1/pow(2,16)
                    mid = torch.where(xx < 0, -1*mid, mid)
                    values.append(mid)
                else:
                    lower_bound = (1.0625 + 0.125 * (j - 1)) / pow(2, 16 - e)
                    upper_bound = (1.0625 + 0.125 * j) / pow(2, 16 - e)
                    mid = (lower_bound+upper_bound) / 2
                    condition = torch.logical_and(x > lower_bound, x < upper_bound)
                    conditions.append(condition)
                    mid = torch.where(xx < 0, -1*mid, mid)
                    values.append(mid)
        return conditions, values

    def conditions_g1(self, xx):
        conditions, values = [], []
        x = torch.where(xx < 0, -1*xx, xx)
        for e in range(16):
            for j in range(8):
                if(j==8): 
                    conditions.append(x > 1.8125*(pow(2,15)))
                    mid = 1.875*(pow(2,15))
                    mid = torch.where(xx < 0, -1*mid, mid)
                    values.append(mid)
                else:
                    lower_bound = (1.0625 + 0.125 * (j - 1)) * pow(2, e)
                    upper_bound = (1.0625 + 0.125 * j) * pow(2, e)
                    mid = (lower_bound+upper_bound) / 2
                    condition = torch.logical_and(x > lower_bound, x < upper_bound)
                    conditions.append(condition)
                    mid = torch.where(xx < 0, -1*mid, mid)
                    values.append(mid)
        return conditions, values
    def forward(self, input):
        if self.num_format == 'fp32':
            return F.conv3d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        elif self.num_format == 'posit':
            # print('going')
            conditions_l1, values_l1 = self.conditions_l1(input)
            conditions_g1, values_g1 = self.conditions_g1(input)
            conditions_l1_w, values_l1_w = self.conditions_l1(self.weight)
            conditions_g1_w, values_g1_w = self.conditions_g1(self.weight)
            for i in range(len(conditions_l1)):
                input = torch.where(torch.abs(input) < 1, torch.where(conditions_l1[i], values_l1[i], input), torch.where(conditions_g1[i], values_g1[i], input))
            for i in range(len(conditions_l1_w)):
                self.weight = torch.nn.Parameter(torch.where(torch.abs(self.weight) < 1, torch.where(conditions_l1_w[i], values_l1_w[i], self.weight), torch.where(conditions_g1_w[i], values_g1_w[i], self.weight)))
            return F.conv3d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

def custom_collate(batch):
    filtered_batch = []
    for video, _, label in batch:
        filtered_batch.append((video, label))
    return torch.utils.data.dataloader.default_collate(filtered_batch)

# class CustomDataset(torch.utils.data.Dataset):
#     def __init__(self, subset_paths, n_frames, training=False):
#         self.subset_paths = subset_paths
#         self.n_frames = n_frames
#         self.training = training
    
#     def __len__(self):
#         return 10

#     def __getitem__(self, idx):
#         # Use your FrameGenerator function to load and preprocess data
#         print(idx)
#         data = FrameGenerator(self.subset_paths[idx], self.n_frames, training=self.training)
#         return data

# 1. Load and normalizing the CIFAR10 training and test datasets using ``torchvision``
def prepare_data():
    # Define data preprocessing and augmentation transforms
    # ucf_data_dir = "data/UCF101/UCF101/UCF-101"
    # ucf_label_dir = "data/UCF101/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist"

    ucf_data_subset_dir_train = "UCF101_subset/train"
    ucf_data_subset_dir_test = "UCF101_subset/test"
    ucf_label_dir = "data/UCF101/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist"
    transform = transforms.Compose([
            # TODO: this should be done by a video-level transfrom when PyTorch provides transforms.ToTensor() for video
            # scale in [0, 1] of type float
            transforms.Lambda(lambda x: x / 255.),
            # reshape into (T, C, H, W) for easier convolutions
            transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
            # rescale to the most common size
            transforms.Lambda(lambda x: nn.functional.interpolate(x, (240, 320))),
])

    # Load Kinetics dataset
    train_dataset = UCF101Dataset(ucf_data_subset_dir_train, ucf_label_dir, frames_per_clip=5,
                       step_between_clips=1, transform=transform)
    # print(train_dataset[0])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1, collate_fn=custom_collate)
    
    test_dataset = torchvision.datasets.UCF101(ucf_data_subset_dir_test, ucf_label_dir, frames_per_clip=5,
                       step_between_clips=1, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=custom_collate)
    # train_dir = list(subset_paths['train'].glob('*/*'))
    # print(train_dir)
    # test_dir = list(subset_paths['test'].glob('*/*'))
    # train_dataset = CustomDataset(train_dir, n_frames=10, training=True)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=1, collate_fn=custom_collate)

    # test_dataset = CustomDataset(test_dir, n_frames=10)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8)

    
    # print(f"Total number of train samples: {len(train_dataset)}")
    # print(f"Total number of test samples: {len(test_dataset)}")
    # print(f"Total number of (train) batches: {len(train_loader)}")
    # print(f"Total number of (test) batches: {len(test_loader)}")
    return train_dataset, train_loader, test_dataset, test_loader

# class BasicBlock3D(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock3D, self).__init__()
#         self.conv1 = CustomConv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm3d(planes)
#         self.conv2 = CustomConv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm3d(planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != planes:
#             self.shortcut = nn.Sequential(
#                 CustomConv3d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm3d(planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out

# # Define the 3D ResNet architecture
# class ResNet3D(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=101):
#         super(ResNet3D, self).__init__()
#         self.in_planes = 64

#         self.conv1 = CustomConv3d(5, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm3d(64)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.linear = nn.Linear(512 * block.expansion, num_classes)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = F.avg_pool3d(out, (1,3,3))
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out

def get_inplanes():
    return [64, 128, 256, 512]

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = CustomConv3d(in_planes, planes, kernel_size=1,stride=1)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = CustomConv3d(planes, planes, kernel_size=1)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = CustomConv3d(in_planes, planes, kernel_size=1)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = CustomConv3d(planes, planes,kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = CustomConv3d(planes, planes * self.expansion, kernel_size=1)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=5,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=101):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = CustomConv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, CustomConv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    CustomConv3d(self.in_planes, planes * block.expansion,kernel_size=1, stride=1),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
# Define a function to create a ResNet3D model
def ResNet183D():
    return ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes())  # You can adjust the number of blocks as needed

def train(net, trainset, trainloader, testset, testloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print("Device: ", device)
    if torch.cuda.is_available():
        if torch.cuda.device_count() >= 1:
            print("Training on", torch.cuda.device_count(), "GPUs!")
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # powersgd = initpowersgd(net)

    
    for epoch in trange(1, desc='epoch'):
        correct = 0
        total = 0
        pass
        running_loss = 0.0
        for i, data in enumerate(tqdm(trainloader, desc='iteration'), 0):
            pass
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            optimizer.step()
            # optimizer_step(optimizer, powersgd)
            running_loss += loss.item()

            if i % 2000 == 1999:    # print every 2000 mini-batches
                tqdm.write('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        print('The training accuracy: %d %%' % (
            100 * correct / total))
    print('Finished Training')


    torch.save(net.state_dict(), PATH)

def test_model(net, trainset, trainloader, testset, testloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    dataiter = iter(testloader)
    # images, labels = dataiter.next()
    images, labels = next(dataiter) #added this instead of line 219
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    net = nn.DataParallel(net) # added this to resolve issue of missing key(s)
    net.load_state_dict(torch.load(PATH))
    net.to(device)
    outputs = net(images.to(device))
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

    # How the network performs on the whole dataset.
    correct = 0
    total = 0
    print('The accuracy on the test dataset is being calculated...')
    with torch.no_grad():
        for data in tqdm(testloader, desc='test iteration'):
            pass
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('The accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

def main():
    print(num_format)
    trainset, trainloader, testset, testloader = prepare_data()
    net = ResNet183D()
    train(net, trainset, trainloader, testset, testloader)
    test_model(net, trainset, trainloader, testset, testloader)

if __name__ == "__main__":
    main()
