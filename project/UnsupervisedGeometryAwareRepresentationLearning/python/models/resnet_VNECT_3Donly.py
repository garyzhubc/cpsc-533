import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F

#import sys
#sys.path.insert(0,'../')

from utils import training as utils_train
from torch.autograd import Variable

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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

def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)

class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=3):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

        self.num_capsules = num_capsules

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
                 range(num_capsules)])

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        device = x.device
        if self.num_route_nodes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]

            logits = Variable(torch.zeros(*priors.size())).to(device)
            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)

        return outputs


class ResNetTwoStream(nn.Module):

    def __init__(self, block, layers, input_key='img_crop', output_keys=['3D'],
                 num_scalars=1000, input_width=256, num_classes=17*3):
        self.output_keys = output_keys
        self.input_key = input_key
        
        self.inplanes = 64
        super(ResNetTwoStream, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block,  64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_reg = self._make_layer(block, 256, layers[3], stride=1)

        self.l4_reg_toVec = nn.Sequential(
                        nn.Conv2d(256* block.expansion, 512, kernel_size=3, stride=1, padding=0, bias=True),
                        nn.BatchNorm2d(512),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(512, 128, kernel_size=5, stride=2, padding=0, bias=False),
                        nn.BatchNorm2d(128),
                        nn.Sigmoid(),
            )
         
        # size computation of fc input: /16 in resnet, /2 in toMap, -3 in map since no padding but 3x3 (-1) and 5x5 (-2) kernels
        l4_vec_width     = int(input_width/32)-3 
        l4_vec_dimension = 128*l4_vec_width*l4_vec_width
        heat_vec_width     = int(input_width/32)-3
        heat_vec_dimension = 128*heat_vec_width*heat_vec_width

        # self.fc = nn.Linear(l4_vec_dimension, num_classes)
        # self.fc = nn.Linear(160, num_classes)
        self.fc = nn.Linear(160, 256)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.conv1a = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=13, stride=1)
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                             kernel_size=9, stride=2)
        self.digit_capsules = CapsuleLayer(num_capsules=10, num_route_nodes=32 * 6 * 6, in_channels=8,
                                           out_channels=16)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x_dict):
        x = x_dict[self.input_key]
        device = x.device
        x = self.conv1(x) # size /2
        x = self.bn1(x)
        x = self.relu(x) 
        x = self.maxpool(x) # size /2

        x = self.layer1(x)

        x = F.relu(self.conv1a(x), inplace=True)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x)

        # x = self.layer2(x)# size /2
        # x = self.layer3(x)# size /2

        x = x.squeeze().transpose(0, 1)
        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)

        _, max_length_indices = classes.max(dim=1)
        y = Variable(torch.eye(10)).to(device).index_select(dim=0, index=max_length_indices.data)
        z = x * y[:, :, None]

        # regression stream
        # x_r = self.layer4_reg(x)
        
        # f_r = self.l4_reg_toVec(x_r)
        # f_lin = f_r.view(f_r.size(0), -1) # 1D per batch
               
        # p = self.fc(f_lin)
        p = self.fc(z.reshape(32, -1))

        #print('f_lin.size()',f_lin.size())
        return {self.output_keys[0]: p} #{'3D': p, '2d_heat': h}


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
#    model = ResNet(Bottleneck, [3, 4, 6, 1], **kwargs)
    model = ResNetTwoStream(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        print("Loading image net weights...")
        utils_train.transfer_partial_weights(model_zoo.load_url(model_urls['resnet50']), model)
        print("Done image net weights...")
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetTwoStream(Bottleneck, [3, 4, 23, 1], **kwargs)
    if pretrained:
        print("Loading image net weights...")
        utils_train.transfer_partial_weights(model_zoo.load_url(model_urls['resnet101']), model)
        print("Done image net weights...")
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetTwoStream(Bottleneck, [3, 8, 36, 1], **kwargs)
    if pretrained:
        print("Loading image net weights...")
        utils_train.transfer_partial_weights(model_zoo.load_url(model_urls['resnet152']), model)
        print("Done image net weights...")
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
