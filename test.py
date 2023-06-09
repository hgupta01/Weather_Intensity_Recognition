import os
import argparse
import torch
from ops.models import TSN
from ops.transforms import *
from main import create_train_transform
from ops.dataset import TSNDataSet
from ops import dataset_config

# import pandas as pd
# import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix,
                             ConfusionMatrixDisplay,
                             classification_report)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def parse_commands():
    parser = argparse.ArgumentParser(
        description="PyTorch implementation of Temporal Segment Networks")

    parser.add_argument('--num_segments', type=int, default=3)
    parser.add_argument('--imgsz', default=224, type=int,
                        help='input image size')
    parser.add_argument('--weights', type=str, default="imagenet")
    parser.add_argument('--weather', type=str, default=None)
    parser.add_argument('--arch', type=str, default="resnet18")
    args = parser.parse_args()
    return args


def load_trained_model(args):
    net = TSN(3, args.num_segments, "RGB",
              base_model="resnet18",
              consensus_type="avg",
              img_feature_dim=256,
              pretrain=False,
              is_shift=True, shift_div=8, shift_place="blockres",
              non_local=False, imgsz=args.imgsz
              )

    checkpoint_path = f"checkpoint/TSM_custom_RGB_{args.arch}_shift8_blockres_avg_segment{args.num_segments}_e50_{args.weather}_{args.weights}_frames{args.num_segments}_imgsz{args.imgsz}/ckpt.best.pth.tar"
    try:
        checkpoint = torch.load(checkpoint_path)
    except:
        print(f"The pretrained model is not present at {checkpoint_path}.")

    checkpoint = checkpoint['state_dict']
    net.load_state_dict(checkpoint)
    return net


def create_dataloader(args):
    cropping = torchvision.transforms.Compose([
        GroupScale(int(args.imgsz*(489/456))),
        GroupCenterCrop(args.imgsz),
    ])

    dataset = TSNDataSet("dataset/videos/frames/", f"dataset/labels/{args.weather}_test.txt", num_segments=args.num_segments,
                         new_length=1,
                         modality="RGB",
                         image_tmpl="{:03d}.jpg",
                         test_mode=True,
                         transform=torchvision.transforms.Compose([
                             cropping,
                             Stack(roll=False),
                             ToTorchFormatTensor(div=True),
                             GroupNormalize(MEAN, STD)])
                         )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True,
    )
    return data_loader


if __name__ == '__main__':
    args = parse_commands()
    net = load_trained_model(args=args)
    dataloader = create_dataloader(args=args)

    print(net)

    true = []
    predict = []

    net.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader):
            logits = net(imgs.cpu())
            true.extend(labels.flatten().tolist())
            predict.extend(logits.softmax(dim=1).cpu(
            ).detach().numpy().argmax(axis=-1).tolist())

    plt.figure(figsize=(8, 6), dpi=300)
    conf_matrix = confusion_matrix(y_true=true, y_pred=predict)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot()
    plt.show()

    print(classification_report(true, predict,
          target_names=["Clear", "Moderate", "Heavy"]))
