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


def calc_loss(model, data_loader, device):
    with torch.no_grad():
        loss = 0
        data_num = 0
        model.eval()
        for minibatch in data_loader:
            x, y = minibatch
            x, y = x.to(device), y.to(device)
            reconstruct = model.forward(x)
            curr_loss = torch.nn.functional.mse_loss(reconstruct, x)
            loss += curr_loss.item() * x.shape[0]
            data_num += 1

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
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--saved_model_path", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--data_num_of_imbalanced_class", type=int, default=2500)
    parser.add_argument("--copy_imbalanced_class", action="store_true")
    parser.add_argument("--use_mixup", action="store_true")
    parser.add_argument("--mixup_alpha", type=float, default=1.0)
    args = parser.parse_args()

    # prepare data_loader
    # transform_normal = torchvision.transforms.Compose(
    #     [torchvision.transforms.ToTensor(),
    #      torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = Dataset()
    train_size = int(len(trainset) * 0.9)
    valid_size = len(trainset) - train_size
    print(train_size, valid_size)
    trainset, validset = torch.utils.data.random_split(trainset, [train_size, valid_size])
    # validset.transform = transform_normal
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    validloader = torch.utils.data.DataLoader(validset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # create model
    model = CNNModel(IMAGE_WIDTH, IMAGE_CHANNEL, args.hidden_size)
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
            if args.use_mixup:
                x, y_a, y_b, lam = mixup_data(x, y, alpha=args.mixup_alpha)
                y_a, y_b = y_a.to(device), y_b.to(device)
            x, y = x.to(device), y.to(device)
            output = model.forward(x)
            loss = torch.nn.functional.mse_loss(output, x, reduction="none").mean([1, 2, 3])

            loss = loss.mean()

            elapsed = time.time() - start
            loss_str = f"{elapsed:.1f}\t{epoch + 1}\t{step + 1}\t{loss:.4f}\t{loss:.4f}"
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

    # save validation loss
    valid_df.to_csv("./result/loss_log/validation_loss.tsv", sep="\t")

    # plot validation loss
    valid_df.plot(x="epoch", y=['loss'], subplots=True, marker=".", figsize=(16, 9))
    plt.savefig('./result/loss_log/validation_loss.png', bbox_inches="tight", pad_inches=0.05)
    plt.clf()
    valid_df.plot(x="epoch", y=[f'accuracy_of_class{i}' for i in range(IMAGE_CHANNEL)], marker=".", figsize=(16, 9))
    plt.savefig('./result/loss_log/accuracy_for_each_class.png', bbox_inches="tight", pad_inches=0.05)

    # save test loss
    with open("./result/loss_log/test_loss.txt", "w") as f:
        test_loss_sum, test_loss_mse, test_loss_ce, test_accuracy, test_accuracy_for_each_class = calc_loss(model, testloader, device, args)
        f.write("loss_sum\tloss_mse\tloss_ce\taccuracy")
        f.write("\n")

        f.write(f"{test_loss_sum:.4f}\t{test_loss_mse:.4f}\t{test_loss_ce:.4f}\t{test_accuracy * 100:.1f}")
        f.write("\n")

    # show reconstruction
    result_image_dir = "./result/image/"
    with torch.no_grad():
        model.eval()
        for minibatch in testloader:
            x, y = minibatch
            x, y = x.to(device), y.to(device)
            out, _ = model.forward(x)

            x = (x + 1) / 2 * 256
            x = x.to(torch.uint8)

            out = (out + 1) / 2 * 256
            out = out.to(torch.uint8)

            for i in range(args.batch_size):
                origin = x[i].reshape([IMAGE_CHANNEL, IMAGE_WIDTH, IMAGE_WIDTH])
                origin = origin.permute([1, 2, 0])
                origin = origin.cpu().numpy()

                pred = out[i].reshape([IMAGE_CHANNEL, IMAGE_WIDTH, IMAGE_WIDTH])
                pred = pred.permute([1, 2, 0])
                pred = pred.cpu().numpy()

                pil_img0 = Image.fromarray(origin)
                pil_img0.save(f"{result_image_dir}/{i}-0.png")
                pil_img1 = Image.fromarray(pred)
                pil_img1.save(f"{result_image_dir}/{i}-1.png")
            exit()


if __name__ == "__main__":
    main()
