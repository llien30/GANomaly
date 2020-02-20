import torch

import yaml
from addict import Dict
import os
import argparse

from net import NetD, NetG
from data import load_data

from libs.weights import weights_init
from model import train

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

if not args.no_wandb:
    wandb.init(
        config=CONFIG,
        name=CONFIG.name,
        project="GANomaly_mnist",  # have to change when you want to change project
        # jobtype="training",
    )

train_dataloader = load_data(CONFIG)["train"]
test_dataloader = load_data(CONFIG)["test"]


G = NetG(CONFIG)
D = NetD(CONFIG)

G.apply(weights_init)
D.apply(weights_init)

if not args.no_wandb:
    # Magic
    wandb.watch(G, log="all")
    wandb.watch(D, log="all")


G_update, D_update = train(
    G,
    D,
    z_dim=CONFIG.z_dim,
    dataloader=train_dataloader,
    CONFIG=CONFIG,
    no_wandb=args.no_wandb,
)

if not os.path.exists(CONFIG.save_dir):
    os.makedirs(CONFIG.save_dir)

torch.save = (
    G_update.state_dict(),
    os.path.join(CONFIG.save_dir, "G", CONFIG.name, ".prm"),
)
torch.save = (
    D_update.state_dict(),
    os.path.join(CONFIG.save_dir, "D", CONFIG.name, ".prm"),
)

print("Done")
