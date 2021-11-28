#!/usr/bin/env python3

import os
import torch
import torchvision
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from model import CNNModel
from dataset import Dataset
from constant import *
from unet_model import UNet


def save_tensor_as_image(x, y, output, save_dir):
    x = x.cpu()
    y = y.cpu()
    output = output.cpu()
    for i in range(x.shape[0]):
        x_image = torchvision.transforms.functional.to_pil_image(x[i])
        y_image = torchvision.transforms.functional.to_pil_image(y[i])
        output_image = torchvision.transforms.functional.to_pil_image(output[i])
        os.makedirs(save_dir, exist_ok=True)
        x_image.save(f"{save_dir}/x{i}.png")
        y_image.save(f"{save_dir}/y{i}.png")
        output_image.save(f"{save_dir}/output{i}.png")


def calc_loss(model, data_loader, device):
    with torch.no_grad():
        loss = 0
        data_num = 0
        model.eval()
        for minibatch in data_loader:
            x, y = minibatch
            x, y = x.to(device), y.to(device)
            output = model.forward(x)
            curr_loss = torch.nn.functional.mse_loss(output, y)
            loss += curr_loss.item() * x.shape[0]
            data_num += x.shape[0]

            IMAGE_SAVE_PATH = "./result/valid_image/"
            save_tensor_as_image(x, y, output, IMAGE_SAVE_PATH)

        loss /= data_num
    return loss


def mixup_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--saved_model_path", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    args = parser.parse_args()

    # prepare data_loader
    trainset = Dataset()
    train_size = int(len(trainset) * 0.9)
    valid_size = len(trainset) - train_size
    print(train_size, valid_size)
    trainset, validset = torch.utils.data.random_split(trainset, [train_size, valid_size])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    validloader = torch.utils.data.DataLoader(validset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # create model
    model = CNNModel(IMAGE_WIDTH, IMAGE_CHANNEL, args.hidden_size)
    model = UNet(1, 1)
    if args.saved_model_path is not None:
        model.load_state_dict(torch.load(args.saved_model_path))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # optimizer
    optim = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epoch)

    # log
    train_df = pd.DataFrame(columns=['time(seconds)', 'epoch', 'loss'])
    valid_df = pd.DataFrame(columns=['time(seconds)', 'epoch', 'loss'])
    start = time.time()
    best_loss = float("inf")

    # training step
    for epoch in range(args.epoch):
        # train
        model.train()
        for step, minibatch in enumerate(trainloader):
            x, y = minibatch
            x, y = x.to(device), y.to(device)
            output = model.forward(x)
            save_tensor_as_image(x, y, output, "./result/train_image")
            loss = torch.nn.functional.mse_loss(output, y, reduction="none").mean([1, 2, 3])
            loss = loss.mean()

            elapsed = time.time() - start
            loss_str = f"{elapsed:.1f}\t{epoch + 1}\t{step + 1}\t{loss:.4f}"
            series = pd.Series([elapsed, int(epoch + 1), loss.item()], index=train_df.columns)
            train_df = train_df.append(series, ignore_index=True)
            os.makedirs(os.path.dirname(TRAIN_LOSS_SAVE_PATH), exist_ok=True)
            train_df.to_csv(TRAIN_LOSS_SAVE_PATH, sep="\t")
            print(loss_str, end="\r")

            optim.zero_grad()
            loss.backward()
            optim.step()

        # validation
        valid_loss = calc_loss(model, validloader, device)
        elapsed = time.time() - start
        series = pd.Series([elapsed, int(epoch + 1), valid_loss], index=valid_df.columns)
        valid_df = valid_df.append(series, ignore_index=True)
        os.makedirs(os.path.dirname(VALID_LOSS_SAVE_PATH), exist_ok=True)
        valid_df.to_csv(VALID_LOSS_SAVE_PATH, sep="\t")
        loss_str = f"{elapsed:.1f}\t{epoch + 1}\t{valid_loss:.4f}"
        print(" " * 100, end="\r")
        print(loss_str)

        if valid_loss < best_loss:
            best_loss = valid_loss
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

        scheduler.step()

    # load best model
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    # plot validation loss
    valid_df.plot(x="epoch", y=['loss'], subplots=True, marker=".", figsize=(16, 9))
    plt.savefig('./result/loss_log/valid_loss.png', bbox_inches="tight", pad_inches=0.05)
    plt.clf()


if __name__ == "__main__":
    main()
