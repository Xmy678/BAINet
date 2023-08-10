import torch
import torch.nn as nn
import torchvision.models as models
from ResNet import ResNet50
from torch.nn import functional as F


print("load BAImodel")

class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                                         kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class GCM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GCM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),

            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


def softmax_2d(x):
    return torch.exp(x) / torch.sum(torch.sum(torch.exp(x), dim=-1, keepdim=True), dim=-2, keepdim=True)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=rate, padding=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.rate = rate

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


class BAINet(nn.Module):
    def __init__(self, channel=512):
        super(BAINet, self).__init__()

        # Backbone model
        self.resnet = ResNet50('rgb')
        self.resnet_depth = ResNet50('rgbd')
        self.rfb3_1 = GCM(1024, channel)
        self.rfb4_1 = GCM(2048, channel)

        self.rbf1 = GCM(2048, channel)
        self.rbf2 = GCM(2048, channel)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 图片尺寸减半
        self.conv1 = BasicConv2d(1024, 512, 3, padding=1)
        self.conv2 = BasicConv2d(1024, 512, 3, padding=1)
        # upsample function
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1_channel1 = nn.Conv2d(64, 64, 1, bias=True)
        self.conv1_spatial1 = nn.Conv2d(64, 1, 3, 1, 1, bias=True)

        self.layer1_channel1 = nn.Conv2d(256, 256, 1, bias=True)
        self.layer1_spatial1 = nn.Conv2d(256, 1, 3, 1, 1, bias=True)

        self.layer2_channel1 = nn.Conv2d(512, 512, 1, bias=True)
        self.layer2_spatial1 = nn.Conv2d(512, 1, 3, 1, 1, bias=True)

        self.layer3_channel1 = nn.Conv2d(1024, 1024, 1, bias=True)
        self.layer3_spatial1 = nn.Conv2d(1024, 1, 3, 1, 1, bias=True)

        self.layer4_channel1 = nn.Conv2d(2048, 2048, 1, bias=True)
        self.layer4_spatial1 = nn.Conv2d(2048, 1, 3, 1, 1, bias=True)

        self.conv1_channel2 = nn.Conv2d(64, 64, 1, bias=True)
        self.conv1_spatial2 = nn.Conv2d(64, 1, 3, 1, 1, bias=True)

        self.layer1_channel2 = nn.Conv2d(256, 256, 1, bias=True)
        self.layer1_spatial2 = nn.Conv2d(256, 1, 3, 1, 1, bias=True)

        self.layer2_channel2 = nn.Conv2d(512, 512, 1, bias=True)
        self.layer2_spatial2 = nn.Conv2d(512, 1, 3, 1, 1, bias=True)

        self.layer3_channel2 = nn.Conv2d(1024, 1024, 1, bias=True)
        self.layer3_spatial2 = nn.Conv2d(1024, 1, 3, 1, 1, bias=True)

        self.layer4_channel2 = nn.Conv2d(2048, 2048, 1, bias=True)
        self.layer4_spatial2 = nn.Conv2d(2048, 1, 3, 1, 1, bias=True)

        self.decoderh = nn.Sequential(

            nn.Dropout(0.5),
            TransBasicConv2d(512, 512, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.Sh = nn.Conv2d(512, 1, 3, stride=1, padding=1)

        self.decoder3 = nn.Sequential(
            BasicConv2d(1024, 512, 3, padding=1),
            BasicConv2d(512, 512, 3, padding=1),
            BasicConv2d(512, 256, 3, padding=1),
            nn.Dropout(0.5),
            TransBasicConv2d(256, 256, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.S3 = nn.Conv2d(256, 1, 3, stride=1, padding=1)
        self.decoder2 = nn.Sequential(
            BasicConv2d(512, 256, 3, padding=1),
            BasicConv2d(256, 256, 3, padding=1),
            BasicConv2d(256, 128, 3, padding=1),
            BasicConv2d(128, 64, 3, padding=1),
            #             BasicConv2d(128, 64, 3, padding=1),
            nn.Dropout(0.5),
            TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.S2 = nn.Conv2d(64, 1, 3, stride=1, padding=1)

        self.decoder1 = nn.Sequential(
            BasicConv2d(128, 64, 3, padding=1),
            BasicConv2d(64, 64, 3, padding=1),
            BasicConv2d(64, 32, 3, padding=1),
            nn.Dropout(0.5),
            TransBasicConv2d(32, 32, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )


        self.S1 = nn.Conv2d(32, 1, 3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

        if self.training:
            self.initialize_weights()

    def bi_attention(self, img_feat, depth_feat, channel_conv1, spatial_conv1, channel_conv2, spatial_conv2):


        img_att = self.avg_pool(img_feat)
        img_att = channel_conv1(img_att)
        img_att = nn.Softmax(dim=1)(img_att) * img_att.shape[1]
        depth_att = self.avg_pool(depth_feat)
        depth_att = channel_conv2(depth_att)
        depth_att = nn.Softmax(dim=1)(depth_att) * depth_att.shape[1]

        img_att = img_att + img_att * depth_att
        depth_att = depth_att + img_att * depth_att

        ca_attentioned_img_feat = img_att * img_feat
        ca_attentioned_depth_feat = depth_att * depth_feat


        img_att1 = F.sigmoid(spatial_conv1(ca_attentioned_img_feat))
        depth_att1 = F.sigmoid(spatial_conv2(ca_attentioned_depth_feat))

        img_att1 = img_att1 + img_att1 * depth_att1
        depth_att1 = depth_att1 + img_att1 * depth_att1

        img_att1 = ca_attentioned_img_feat * img_att1
        depth_att1 = ca_attentioned_depth_feat * depth_att1

        return img_att1, depth_att1

    def forward(self, x, x_depth):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)



        x_depth = self.resnet_depth.conv1(x_depth)
        x_depth = self.resnet_depth.bn1(x_depth)
        x_depth = self.resnet_depth.relu(x_depth)


        tempf, temp = self.bi_attention(x, x_depth, self.conv1_channel1, self.conv1_spatial1, self.conv1_channel2,
                                        self.conv1_spatial2)


        f1 = tempf + temp + tempf.mul(temp)


        x = self.resnet.maxpool(tempf)

        x1 = self.resnet.layer1(x)


        x_depth = self.resnet_depth.maxpool(temp)

        x1_depth = self.resnet_depth.layer1(x_depth)


        tempf, temp = self.bi_attention(x1, x1_depth, self.layer1_channel1, self.layer1_spatial1, self.layer1_channel2,
                                        self.layer1_spatial2)

        f2 = tempf + temp + tempf.mul(temp)

        x2 = self.resnet.layer2(tempf)
        x2_depth = self.resnet_depth.layer2(temp)
        tempf, temp = self.bi_attention(x2, x2_depth, self.layer2_channel1, self.layer2_spatial1, self.layer2_channel2,
                                        self.layer2_spatial2)

        f3 = tempf + temp + tempf.mul(temp)

        x3 = self.resnet.layer3_1(tempf)
        x3_depth = self.resnet_depth.layer3_1(temp)
        tempf, temp = self.bi_attention(x3, x3_depth, self.layer3_channel1, self.layer3_spatial1, self.layer3_channel2,
                                        self.layer3_spatial2)

        f4 = tempf + temp + tempf.mul(temp)

        x4 = self.resnet.layer4_1(tempf)
        x4_depth = self.resnet_depth.layer4_1(temp)
        tempf, temp = self.bi_attention(x4, x4_depth, self.layer4_channel1, self.layer4_spatial1, self.layer4_channel2,
                                        self.layer4_spatial2)

        f5 = tempf + temp + tempf.mul(temp)

        x5 = self.rbf1(tempf)
        x5_depth = self.rbf2(temp)
        f6 = x5 + x5_depth


        f4 = self.rfb3_1(f4)
        f5 = self.rfb4_1(f5)


        ft = nn.Softmax(dim=1)(f6) * f5


        ft = torch.cat((f6, ft), 1)
        ft = self.conv1(ft)
        fh = nn.Softmax(dim=1)(f6 + f5) * f4
        fh = torch.cat((ft, fh), 1)
        fh = self.conv2(fh)


        fh_up = self.decoderh(fh)
        sh = self.Sh(fh_up)
        f3_up = self.decoder3(
            torch.cat((f3, fh_up), 1))

        s3 = self.S3(f3_up)

        f2_up = self.decoder2(torch.cat((f2, f3_up), 1))

        s2 = self.S2(f2_up)

        f1_up = self.decoder1(torch.cat((f1, f2_up), 1))

        s1 = self.S1(f1_up)

        s2 = self.upsample2(s2)
        s3 = self.upsample4(s3)
        sh = self.upsample8(sh)

        return s1, s2, s3, sh, self.sigmoid(s1), self.sigmoid(s2), self.sigmoid(s3), self.sigmoid(sh)

    # # initialize the weights
    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)

        all_params = {}
        for k, v in self.resnet_depth.state_dict().items():
            if k == 'conv1.weight':
                all_params[k] = torch.nn.init.normal_(v, mean=0, std=1)
            elif k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet_depth.state_dict().keys())
        self.resnet_depth.load_state_dict(all_params)

