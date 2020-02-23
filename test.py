import torch
from torchvision.utils import save_image

import yaml
from addict import Dict
import os
import argparse

from net import NetG
from data import load_data
from lib.roc import roc

import time

import wandb


def get_parameters():
    """
    make parser to get parameters
    """
    parser = argparse.ArgumentParser(description="take config file path")
    parser.add_argument("config", type=str, help="path of a config file for training")
    parser.add_argument("--no_wandb", action="store_true", help="Add --no_wandb option")

    return parser.parse_args()


args = get_parameters()

CONFIG = Dict(yaml.safe_load(open(args.config)))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not args.no_wandb:
    wandb.init(
        config=CONFIG,
        name=CONFIG.name,
        project="GANomaly_mnist",  # have to change when you want to change project
        # jobtype="training",
    )

# load weights of G
path = os.path.join(CONFIG.save_dir, "G", CONFIG.name, ".prm")
pretrained_dict = torch.load(path)["state_dict"]

try:
    NetG.load_state_dict(pretrained_dict)
except IOError:
    raise IOError("NetG weights not found")
    print("   Loaded weights.")

test_data = load_data["test"]
# where to save scores
Scores = torch.zeros(
    size=(len(test_data.dataset),), dtype=torch.float32, device=device,
)
# where to save test labels
Labels = torch.zeros(size=(len(test_data.dataset),), dtype=torch.long, device=device)

total_steps = 0
times = []
for i, (imges, label) in enumerate(test_data, 0):
    mini_batch_size = imges.size()[0]
    total_steps = total_steps + imges.size()[0]
    start_time = time.time()

    imges = imges.reshape(-1, CONFIG.channel, CONFIG.input_size, CONFIG.input_size)
    imges = imges.to(device)
    fake_img, latent_i, latent_o = NetG(imges)
    # caliculate the img's error
    error = torch.mean(torch.pow((latent_i - latent_o), 2), dim=1)

    last_time = time.time()

    Scores[i * mini_batch_size : i * mini_batch_size + error.size(0)] = error.reshape(
        error.size(0)
    )
    Labels[i * mini_batch_size : i * mini_batch_size + error.size(0)] = label.reshape(
        error.size(0)
    )

    times.append(last_time - start_time)

    save_path_r = os.join("./test_img" + CONFIG.name + "real" + "{}.png".format(i))
    save_path_f = os.join("./test_img" + CONFIG.name + "fake" + "{}.png".format(i))
    save_image(imges, save_path_r)
    save_image(fake_img, save_path_f)

# Normalize score
Scores = (Scores - torch.min(Scores)) / (torch.max(Scores) - torch.min(Scores))
Acc = roc(Labels, Scores)

if not args.no_wandb:
    wandb.log({"Accuracy": Acc})

print("Abnormal_number-{} || Accuracy(ROC):{}".format(CONFIG.abnormal_class, Acc))
