"""
ReXNet
Copyright (c) 2020-present NAVER Corp.
MIT license
"""

from math import ceil

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torchvision import transforms

from resnext import ReXNetV1
from typing import List, Dict, Tuple
import pandas as pd


# # Memory-efficient Siwsh using torch.jit.script borrowed from the code in (https://twitter.com/jeremyphoward/status/1188251041835315200)
# # Currently use memory-efficient SiLU as default:
# USE_MEMORY_EFFICIENT_SiLU = True
#
# if USE_MEMORY_EFFICIENT_SiLU:
#     @torch.jit.script
#     def silu_fwd(x):
#         return x.mul(torch.sigmoid(x))
#
#
#     @torch.jit.script
#     def silu_bwd(x, grad_output):
#         x_sigmoid = torch.sigmoid(x)
#         return grad_output * (x_sigmoid * (1. + x * (1. - x_sigmoid)))
#
#
#     class SiLUJitImplementation(torch.autograd.Function):
#         @staticmethod
#         def forward(ctx, x):
#             ctx.save_for_backward(x)
#             return silu_fwd(x)
#
#         @staticmethod
#         def backward(ctx, grad_output):
#             x = ctx.saved_tensors[0]
#             return silu_bwd(x, grad_output)
#
#
#     def silu(x, inplace=False):
#         return SiLUJitImplementation.apply(x)
#
# else:
#     def silu(x, inplace=False):
#         return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid())
#
#
# class SiLU(nn.Module):
#     def __init__(self, inplace=True):
#         super(SiLU, self).__init__()
#         self.inplace = inplace
#
#     def forward(self, x):
#         return silu(x, self.inplace)
#
#
# def ConvBNAct(out, in_channels, channels, kernel=1, stride=1, pad=0,
#               num_group=1, active=True, relu6=False):
#     out.append(nn.Conv2d(in_channels, channels, kernel,
#                          stride, pad, groups=num_group, bias=False))
#     out.append(nn.BatchNorm2d(channels))
#     if active:
#         out.append(nn.ReLU6(inplace=True) if relu6 else nn.ReLU(inplace=True))
#
#
# def ConvBNSiLU(out, in_channels, channels, kernel=1, stride=1, pad=0, num_group=1):
#     out.append(nn.Conv2d(in_channels, channels, kernel,
#                          stride, pad, groups=num_group, bias=False))
#     out.append(nn.BatchNorm2d(channels))
#     out.append(SiLU(inplace=True))
#
#
# class SE(nn.Module):
#     def __init__(self, in_channels, channels, se_ratio=12):
#         super(SE, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Conv2d(in_channels, channels // se_ratio, kernel_size=1, padding=0),
#             nn.BatchNorm2d(channels // se_ratio),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels // se_ratio, channels, kernel_size=1, padding=0),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = self.fc(y)
#         return x * y
#
#
# class LinearBottleneck(nn.Module):
#     def __init__(self, in_channels, channels, t, stride, use_se=True, se_ratio=12, dropout=0.0,
#                  **kwargs):
#         super(LinearBottleneck, self).__init__(**kwargs)
#         self.use_shortcut = stride == 1 and in_channels <= channels
#         self.in_channels = in_channels
#         self.out_channels = channels
#         self.drop = nn.Dropout2d(dropout)
#         out = []
#         if t != 1:
#             dw_channels = in_channels * t
#             ConvBNSiLU(out, in_channels=in_channels, channels=dw_channels)
#         else:
#             dw_channels = in_channels
#
#         ConvBNAct(out, in_channels=dw_channels, channels=dw_channels, kernel=3, stride=stride, pad=1,
#                   num_group=dw_channels, active=False)
#
#         if use_se:
#             out.append(SE(dw_channels, dw_channels, se_ratio))
#
#         out.append(nn.ReLU6())
#         ConvBNAct(out, in_channels=dw_channels, channels=channels, active=False, relu6=True)
#         self.out = nn.Sequential(*out)
#
#     def forward(self, x):
#         out = self.out(x)
#         if self.use_shortcut:
#             out[:, 0:self.in_channels] += x
#         out = self.drop(out)
#         return out
#
#
# class ReXNetV1(nn.Module):
#     def __init__(self, input_ch=16, final_ch=180, width_mult=1.0, depth_mult=1.0, classes=1000,
#                  use_se=True,
#                  se_ratio=12,
#                  dropout_ratio=0.2,
#                  dropout_path=0.25,
#                  bn_momentum=0.9):
#         super(ReXNetV1, self).__init__()
#
#         layers = [1, 2, 2, 3, 3, 5]
#         strides = [1, 2, 2, 2, 1, 2]
#         use_ses = [False, False, True, True, True, True]
#
#         layers = [ceil(element * depth_mult) for element in layers]
#         strides = sum([[element] + [1] * (layers[idx] - 1)
#                        for idx, element in enumerate(strides)], [])
#         if use_se:
#             use_ses = sum([[element] * layers[idx] for idx, element in enumerate(use_ses)], [])
#         else:
#             use_ses = [False] * sum(layers[:])
#         ts = [1] * layers[0] + [6] * sum(layers[1:])
#
#         self.depth = sum(layers[:]) * 3
#         stem_channel = 32 / width_mult if width_mult < 1.0 else 32
#         inplanes = input_ch / width_mult if width_mult < 1.0 else input_ch
#
#         features = []
#         in_channels_group = []
#         channels_group = []
#
#         # The following channel configuration is a simple instance to make each layer become an expand layer.
#         for i in range(self.depth // 3):
#             if i == 0:
#                 in_channels_group.append(int(round(stem_channel * width_mult)))
#                 channels_group.append(int(round(inplanes * width_mult)))
#             else:
#                 in_channels_group.append(int(round(inplanes * width_mult)))
#                 inplanes += final_ch / (self.depth // 3 * 1.0)
#                 channels_group.append(int(round(inplanes * width_mult)))
#
#         ConvBNSiLU(features, 3, int(round(stem_channel * width_mult)), kernel=3, stride=2, pad=1)
#
#         for block_idx, (in_c, c, t, s, se) in enumerate(zip(in_channels_group, channels_group, ts, strides, use_ses)):
#             features.append(LinearBottleneck(in_channels=in_c,
#                                              channels=c,
#                                              t=t,
#                                              stride=s,
#                                              use_se=se, se_ratio=se_ratio, dropout=dropout_path))
#
#         pen_channels = int(1280 * width_mult)
#         ConvBNSiLU(features, c, pen_channels)
#
#         features.append(nn.AdaptiveAvgPool2d(1))
#         self.features = nn.Sequential(*features)
#         # self.output = nn.Sequential(
#         #     nn.Dropout(dropout_ratio),
#         #     nn.Conv2d(pen_channels, classes, 1, bias=True))
#         self.out = nn.Sequential(
#             nn.Dropout(dropout_ratio),
#             nn.Conv2d(pen_channels, classes, 1, bias=True))
#
#     def extract_features(self, x):
#         return self.features[:-1](x)
#
#     def forward(self, x):
#         x = self.features(x)
#         x = self.out(x).flatten(1)
#         return x


# 224, False, None
def build_transform(input_size, imagenet_default_mean_and_std, crop_pct):
    resize_im = input_size > 32
    # input_size = input_size
    # imagenet_default_mean_and_std = imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if input_size >= 384:
            t.append(
                transforms.Resize((input_size, input_size),
                                  interpolation=transforms.InterpolationMode.BICUBIC),
            )
            print(f"Warping {input_size} size input images...")
        else:
            if crop_pct is None:
                crop_pct = 224 / 256
            size = int(input_size / crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            )
            t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


# @torch.no_grad()
# def evaluate(model, device):
#
#     # switch to evaluation mode
#     model.eval()
#
#     images = images.to(device, non_blocking=True)
#     output = model(images)
#     _, predicted_class = torch.max(output.data, 1)

# for batch in metric_logger.log_every(data_loader, 10, header):
#     images = batch[0]
#     target = batch[-1]
#     labels_onehot = batch[1]
#
#     images = images.to(device, non_blocking=True)
#     target = target.to(device, non_blocking=True)
#
#     # compute output
#     if use_amp:
#         with torch.cuda.amp.autocast():
#             output = model(images)
#             loss = criterion(output, target)
#     else:
#         output = model(images)
#         loss = criterion(output, target)
#
#     _, predicted_class = torch.max(output.data, 1)


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def auto_load_model(resume, model):
    # output_dir = Path(args.output_dir)
    # if args.auto_resume and len(args.resume) == 0:
    #     import glob
    #     all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
    #     latest_ckpt = -1
    #     for ckpt in all_checkpoints:
    #         t = ckpt.split('-')[-1].split('.')[0]
    #         if t.isdigit():
    #             latest_ckpt = max(int(t), latest_ckpt)
    #     if latest_ckpt >= 0:
    #         args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
    #     print("Auto resume checkpoint: %s" % args.resume)

    if resume is not None:
        # if args.resume.startswith('https'):
        #     checkpoint = torch.hub.load_state_dict_from_url(
        #         args.resume, map_location='cpu', check_hash=True)
        # else:
        checkpoint = torch.load(resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % resume)
        # if 'optimizer' in checkpoint and 'epoch' in checkpoint:
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        #     if not isinstance(checkpoint['epoch'], str): # does not support resuming with 'best', 'best-ema'
        #         args.start_epoch = checkpoint['epoch'] + 1
        #     else:
        #         assert args.eval, 'Does not support resuming with checkpoint-best'
        #     if hasattr(args, 'model_ema') and args.model_ema:
        #         if 'model_ema' in checkpoint.keys():
        #             model_ema.ema.load_state_dict(checkpoint['model_ema'])
        #         else:
        #             model_ema.ema.load_state_dict(checkpoint['model'])
        #     if 'scaler' in checkpoint:
        #         loss_scaler.load_state_dict(checkpoint['scaler'])
        #     print("With optim & sched!")

def inference(file, nb_classes, drop_path, classes):
    file = Image.open(file).convert('RGB')
    # img = transform(file).unsqueeze(0)
    transform = build_transform(224, False, None);
    img = transform(file).unsqueeze(0)

    model = ReXNetV1(width_mult=3.0, classes=nb_classes, dropout_path=drop_path)
    # model.load_state_dict(torch.load('checkpoint-19.pth'), strict=False)
    auto_load_model('checkpoint-19.pth', model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        print('Passing your image to the model....')
        output = model(img.to(device))
        # look, predicted_class = torch.max(output.data, 1)
        outputs_class = softmax(output.data.cpu().numpy())
        probs = np.asarray(outputs_class)
        preds = np.argmax(probs, axis=1)
        # probs2 = np.asarray(probs).flatten()
        value = preds[0]
        print("Predicted Severity Value: ", value)
        print("class is: ", classes[value])
        print('Your image is printed:')
        return value, classes[value]
        # return 2, classes[2]

    # model.to(device)

    # file = Image.open(file).convert('RGB')
    # img = transform(file).unsqueeze(0)
    # print('Transforming your image...')
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.eval()
    # with torch.no_grad():
    #     print('Passing your image to the model....')
    #     out = model(img.to(device))
    #     ps = torch.exp(out)
    #     top_p, top_class = ps.topk(1, dim=1)
    #     value = top_class.item()
    #     print("Predicted Severity Value: ", value)
    #     print("class is: ", classes[value])
    #     print('Your image is printed:')
    #     return value, classes[value]


def resnextmain(path):
    classes = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
    x, y = inference(path, 5, 0.3, classes)
    return x, y

def readCsv(label_dir: str):
    # 读取CSV文件
    data = pd.read_csv(label_dir)

    # 初始化一个字典，用于存储分组数据
    labelValue = []
    grouped_data = []

    # 遍历数据中的每行
    for index, row in data.iterrows():
        image_name = row[0]  # 图像名称
        label = row[1]  # 标签值
        grouped_data.append((image_name, label))
        labelValue.append(label)
    return grouped_data, labelValue



if __name__ == '__main__':
    csvPath = "F:/wei/Retinal_blindness_detection_Pytorch/sampleimages/test2.csv"
    grouped_images, labelValue = readCsv(csvPath)
    preds = []
    for image_name, label in grouped_images:
        image_path = "F:/wei/Retinal_blindness_detection_Pytorch/sampleimages/test_images2/" + image_name + ".png"
        x, y = resnextmain(image_path)
        preds.append(x)
    print(labelValue)
    print(preds)
