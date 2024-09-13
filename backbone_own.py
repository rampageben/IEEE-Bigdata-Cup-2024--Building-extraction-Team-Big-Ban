import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, ResNet101_Weights, Swin_V2_B_Weights, Swin_V2_T_Weights, Swin_V2_S_Weights
from torch import Tensor
from typing import Dict
from timm import create_model
import numpy as np

class torchSwinBackbone(nn.Module):
    def __init__(self, version='swin_base', freeze_at=0):
        super(torchSwinBackbone, self).__init__()
        
        if version == 'swin_base':
            swin = models.swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1)
        elif version == 'swin_tiny':
            swin = models.swin_v2_t(weights=Swin_V2_T_Weights.IMAGENET1K_V1)
        elif version == 'swin_small':
            swin = models.swin_v2_s(weights=Swin_V2_S_Weights.IMAGENET1K_V1)
        else:
            raise ValueError("Unsupported torchSwin version.")
        
        self.feature_channels = {
            'swin_tiny': [96, 192, 384, 768],
            'swin_small': [96, 192, 384, 768],
            'swin_base': [128, 256, 512, 1024],
        }

        self.stage1 = swin.features[0:2]
        self.stage2 = swin.features[2:4]
        self.stage3 = swin.features[4:6]
        self.stage4 = swin.features[6:8]
        
        self.freeze_layers(freeze_at)
        
    def forward(self, x):
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        return [c1, c2, c3, c4]

    def freeze_layers(self, freeze_at):
        if freeze_at >= 1:
            for param in self.stage1.parameters():
                param.requires_grad = False
        if freeze_at >= 2:
            for param in self.stage2.parameters():
                param.requires_grad = False
        if freeze_at >= 3:
            for param in self.stage3.parameters():
                param.requires_grad = False
        if freeze_at >= 4:
            for param in self.stage4.parameters():
                param.requires_grad = False


class ResNetBackbone(nn.Module):
    def __init__(self, version='resnet50', freeze_at=0):
        super(ResNetBackbone, self).__init__()
        
        if version == 'resnet50':
            resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        elif version == 'resnet101':
            resnet = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
        elif version == 'resnext50_32x4d':
            resnet = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
        elif version == 'resnext101_32x8d':
            resnet = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V2)
        else:
            raise ValueError("Unsupported ResNet version. Use 'resnet50', 'resnet101', 'resnext50_32x4d' or 'resnext101_32x8d'")
        
        self.stage1 = nn.Sequential(*list(resnet.children())[:5]) 
        self.stage2 = nn.Sequential(*list(resnet.children())[5]) 
        self.stage3 = nn.Sequential(*list(resnet.children())[6]) 
        self.stage4 = nn.Sequential(*list(resnet.children())[7]) 

        self.freeze_layers(freeze_at)
        
    def forward(self, x):
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        return [c1, c2, c3, c4]

    def freeze_layers(self, freeze_at):
        if freeze_at >= 1:
            for param in self.stage1.parameters():
                param.requires_grad = False
        if freeze_at >= 2:
            for param in self.stage2.parameters():
                param.requires_grad = False
        if freeze_at >= 3:
            for param in self.stage3.parameters():
                param.requires_grad = False
        if freeze_at >= 4:
            for param in self.stage4.parameters():
                param.requires_grad = False


class ConvNeXtBackbone(nn.Module):
    def __init__(self, version='convnext_tiny.fb_in22k_ft_in1k_384', freeze_at=0):
        super(ConvNeXtBackbone, self).__init__()
        
        self.convnext = create_model(version, pretrained=True, features_only=True)
        self.feature_channels = {
            'convnext_tiny.fb_in22k_ft_in1k_384': [96, 192, 384, 768],
            'convnext_small.fb_in22k_ft_in1k_384': [96, 192, 384, 768],
            'convnext_base.fb_in22k_ft_in1k_384': [128, 256, 512, 1024],
            'convnext_large.fb_in22k_ft_in1k_384': [192, 384, 768, 1536],
            'convnextv2_huge.fcmae_ft_in22k_in1k_512': [352, 704, 1408, 2816],
        }

        if version not in self.feature_channels:
            raise ValueError("Unsupported ConvNeXt version. Please use a valid version from timm library.")

        self.freeze_layers(freeze_at)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        features = self.convnext(x)
        return features

    def freeze_layers(self, freeze_at):
        if freeze_at >= 1:
            for param in self.convnext.parameters():
                param.requires_grad = False
        if freeze_at >= 2:
            for param in self.convnext.parameters():
                param.requires_grad = False
        if freeze_at >= 3:
            for param in self.convnext.parameters():
                param.requires_grad = False
        if freeze_at >= 4:
            for param in self.convnext.parameters():
                param.requires_grad = False


class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=[1, 3, 5, 7, 9]):
        super(SPPF, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels // 2, kernel_size=1, stride=1, padding=0)

        self.conv3 = nn.Conv2d(out_channels // 2 * (len(pool_sizes) + 1), out_channels, kernel_size=1, stride=1, padding=0)

        self.pool_sizes = pool_sizes
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)  
                nn.init.constant_(m.bias, 0)    

    def forward(self, feature_maps):
        x = feature_maps[-1]
        
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        pooled_outputs = [x]
        for size in self.pool_sizes:
            pooled = nn.functional.max_pool2d(x, kernel_size=size, stride=1, padding=size//2)
            pooled_outputs.append(pooled)

        x = torch.cat(pooled_outputs, dim=1)
        x = self.conv3(x)

        feature_maps.append(x)

        return feature_maps


def fourier_combine(x1, x2, output_size=(6, 6)):
    F1 = torch.fft.fft2(x1, s=output_size)
    F2 = torch.fft.fft2(x2, s=output_size)
    combined_F = F1 + F2
    combined_matrix = torch.fft.ifft2(combined_F).real

    return combined_matrix


def get_middle_size(size1, size2):
    return tuple((np.array(size1) + np.array(size2) + 1) // 2)


class FPN_FF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPN_FF, self).__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, groups=in_channel),  
                nn.Conv2d(in_channel, out_channels, kernel_size=1)  
            )
            for in_channel in in_channels
        ])
        
        self.concat_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            )
            for _ in range(len(in_channels) - 1)
        ])

        self.down_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            )
            for _ in range(len(in_channels) - 1)
        ])

        self.down_ff_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            )
            for _ in range(len(in_channels) - 2)
        ])

        final_count = 2 * len(in_channels) - 1
        self.concat_pan_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            )
            for _ in range(final_count)
        ])
        
        self._init_weights()

    def _init_weights(self):
        for layer in self.lateral_convs:
            for sublayer in layer:
                if isinstance(sublayer, nn.Conv2d):
                    nn.init.kaiming_normal_(sublayer.weight, mode='fan_out', nonlinearity='leaky_relu')
                    if sublayer.bias is not None:
                        nn.init.constant_(sublayer.bias, 0)
                elif isinstance(sublayer, nn.BatchNorm2d):
                    nn.init.constant_(sublayer.weight, 1)
                    nn.init.constant_(sublayer.bias, 0)
        
        for layer in self.concat_convs:
            for sublayer in layer:
                if isinstance(sublayer, nn.Conv2d):
                    nn.init.kaiming_normal_(sublayer.weight, mode='fan_out', nonlinearity='leaky_relu')
                    if sublayer.bias is not None:
                        nn.init.constant_(sublayer.bias, 0)
                elif isinstance(sublayer, nn.BatchNorm2d):
                    nn.init.constant_(sublayer.weight, 1)
                    nn.init.constant_(sublayer.bias, 0)
        
        for layer in self.down_convs:
            for sublayer in layer:
                if isinstance(sublayer, nn.Conv2d):
                    nn.init.kaiming_normal_(sublayer.weight, mode='fan_out', nonlinearity='leaky_relu')
                    if sublayer.bias is not None:
                        nn.init.constant_(sublayer.bias, 0)
                elif isinstance(sublayer, nn.BatchNorm2d):
                    nn.init.constant_(sublayer.weight, 1)
                    nn.init.constant_(sublayer.bias, 0)
        
        for layer in self.down_ff_convs:
            for sublayer in layer:
                if isinstance(sublayer, nn.Conv2d):
                    nn.init.kaiming_normal_(sublayer.weight, mode='fan_out', nonlinearity='leaky_relu')
                    if sublayer.bias is not None:
                        nn.init.constant_(sublayer.bias, 0)
                elif isinstance(sublayer, nn.BatchNorm2d):
                    nn.init.constant_(sublayer.weight, 1)
                    nn.init.constant_(sublayer.bias, 0)
        
        for layer in self.concat_pan_convs:
            for sublayer in layer:
                if isinstance(sublayer, nn.Conv2d):
                    nn.init.kaiming_normal_(sublayer.weight, mode='fan_out', nonlinearity='leaky_relu')
                    if sublayer.bias is not None:
                        nn.init.constant_(sublayer.bias, 0)
                elif isinstance(sublayer, nn.BatchNorm2d):
                    nn.init.constant_(sublayer.weight, 1)
                    nn.init.constant_(sublayer.bias, 0)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        features = [lateral_conv(feature) for lateral_conv, feature in zip(self.lateral_convs, x)]
        
        for i in range(len(features) - 2, -1, -1):
            if features[i].device.type == 'cuda':
                upsampled_feature = F.interpolate(features[i + 1], size=features[i].shape[-2:], mode='bicubic', align_corners=False)
            else:
                upsampled_feature = F.interpolate(features[i + 1], size=features[i].shape[-2:], mode='bilinear', align_corners=False)
            
            concat_feature = torch.cat([features[i], upsampled_feature], dim=1)
            features[i] = self.concat_convs[i](concat_feature)

        final_features = []
        for i in range(len(features) - 1):
            final_features.append(features[i])
            middle_size = get_middle_size(features[i].shape[-2:], features[i + 1].shape[-2:])
            combined_feature = fourier_combine(features[i], features[i + 1], output_size=middle_size)
            final_features.append(combined_feature)
        final_features.append(features[-1])

        for i in range(2, len(final_features) - 1, 2):
            downsampled_feature = self.down_convs[i // 2 - 1](final_features[i - 2])
            final_features[i] = self.concat_pan_convs[i // 2 - 1](torch.cat([downsampled_feature, final_features[i]], dim=1))

        for i in range(3, len(final_features) - 1, 2):
            down_ff_feature = self.down_ff_convs[i // 2 - 1](final_features[i - 2])
            final_features[i] = self.concat_pan_convs[i // 2 - 1](torch.cat([down_ff_feature, final_features[i]], dim=1))
    
        return {str(i): feature for i, feature in enumerate(final_features)}


class Backbone(nn.Module):
    def __init__(self, backbone_type='resnet', version=None, freeze_at=0, out_channels=256):
        super(Backbone, self).__init__()
        
        if backbone_type == 'resnet':
            if version is None:
                version = 'resnet50'
            self.own_backbone = ResNetBackbone(version=version, freeze_at=freeze_at)
            in_channels = [256, 512, 1024, 2048]
        elif backbone_type == 'convnext':
            if version is None:
                version = 'convnext_tiny'
            self.own_backbone = ConvNeXtBackbone(version=version, freeze_at=freeze_at)
            in_channels = self.own_backbone.feature_channels[version]
        elif backbone_type == 'torchswin':
            if version is None:
                version = 'swin_base'
            self.own_backbone = torchSwinBackbone(version=version, freeze_at=freeze_at)
            in_channels = self.own_backbone.feature_channels[version]
        else:
            raise ValueError("Unsupported backbone type. Use 'resnet', 'swin', or 'convnext'")
        
        self.own_sppf = SPPF(in_channels[-1], in_channels[-1] * 2)
        in_channels.append(in_channels[-1] * 2)
        self.own_fpn_pan = FPN_FF(in_channels, out_channels)
        self.out_channels = out_channels
        self.backbone_type = backbone_type

    def forward(self, x):
        x = self.own_backbone(x)
        if self.backbone_type in ['torchswin']:
            for i in range(len(x)):
                x[i] = x[i].permute(0, 3, 1, 2)
        x = self.own_sppf(x)
        x = self.own_fpn_pan(x)
        
        return x
