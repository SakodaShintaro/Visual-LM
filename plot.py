from constant import *
import pandas as pd
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", type=str, default="./result/loss_log")
args = parser.parse_args()

valid_df = pd.read_csv(f"{args.log_dir}/valid_loss.tsv", sep="\t")

# plot validation loss
valid_df.plot(x="epoch", y=['loss'], subplots=True, marker=".")
plt.xlabel("Learning Epoch")
plt.ylabel("Loss")
plt.savefig(f'{args.log_dir}/valid_loss.png', bbox_inches="tight", pad_inches=0.05)
plt.clf()
