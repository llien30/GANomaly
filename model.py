import torch
import torch.nn as nn

# from torchvision.utils import save_image

from libs.loss import L2_Loss
from libs.meter import AverageMeter

import time

# from PIL import Image

import wandb


def train(G, D, z_dim, dataloader, CONFIG, no_wandb):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    lr = 0.0002
    beta1, beta2 = 0.5, 0.999

    g_optimizer = torch.optim.Adam(G.parameters(), lr, [beta1, beta2])
    d_optimizer = torch.optim.Adam(D.parameters(), lr, [beta1, beta2])

    # the default mini batch size
    mini_batch_size = 64
    # fixed_img = torch.randn(CONFIG.num_fakeimg, 1, 64, 64).to(device)

    G.to(device)
    D.to(device)

    G.train()
    D.train()

    torch.backends.cudnn.benchmark = True

    num_epochs = CONFIG.num_epochs

    for epoch in range(num_epochs):
        t_epoch_start = time.time()

        print("----------------------(train)----------------------")
        print("Epoch {}/{}".format(epoch, num_epochs))
        print("---------------------------------------------------")

        d_loss_meter = AverageMeter("D_loss", ":.4e")
        g_loss_meter = AverageMeter("G_loss", ":.4e")

        for imges in dataloader:
            print(imges.size())
            imges.reshape(
                CONFIG.batch_size, CONFIG.channel, CONFIG.input_size, CONFIG.input_size
            )
            imges = imges.to(device)
            mini_batch_size = imges.size()[0]

            real_label = torch.ones(
                size=(mini_batch_size,), dtype=torch.float32, device=device
            )
            fake_label = torch.zeros(
                size=(mini_batch_size,), dtype=torch.float32, device=device
            )

            """
            Forward Pass
            """
            train_fake_img, latent_input, latent_output = G(imges)
            pred_real, feat_real = D(imges, CONFIG)
            pred_fake, feat_fake = D(train_fake_img, CONFIG)

            """
            Loss Calculation
            """
            L1_Loss = nn.L1Loss()
            # Binary Cross Entropy
            D_Loss = nn.BCEWithLogitsLoss(reduction="mean")

            loss_g_adv = L2_Loss(feat_real, feat_fake)
            loss_g_con = L1_Loss(train_fake_img, imges)
            loss_g_enc = L2_Loss(latent_input, latent_output)

            w_adv = CONFIG.w_adv
            w_con = CONFIG.w_con
            w_enc = CONFIG.w_enc

            g_loss = w_adv * loss_g_adv + w_con * loss_g_con + w_enc * loss_g_enc
            g_loss_meter.update(g_loss.item())

            loss_d_real = D_Loss(pred_real, real_label)
            loss_d_fake = D_Loss(pred_fake, fake_label)

            d_loss = (loss_d_real + loss_d_fake) * 0.5
            d_loss_meter.update(d_loss.item())

            """
            Backward Pass
            """
            g_optimizer.zero_grad()
            g_loss.backward(retain_graph=True)
            g_optimizer.step()

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

        t_epoch_finish = time.time()
        print("---------------------------------------------------")
        print(
            "Epoch{}|| D_Loss :{:.4f} || G_Loss :{:.4f}".format(
                epoch, d_loss_meter.avg, g_loss_meter.avg,
            )
        )
        print("timer:  {:.4f} sec.".format(t_epoch_finish - t_epoch_start))
        # fake_imges = G(fixed_img)
        # save_image(fake_imges, "fake_imges.png")

        if not no_wandb:
            wandb.log(
                {
                    "train_time": t_epoch_finish - t_epoch_start,
                    "d_loss": d_loss_meter.avg,
                    "g_loss": g_loss_meter.avg,
                },
                step=epoch,
            )

            # img = Image.open("fake_imges.png")
            # wandb.log({"image": [wandb.Image(img)]}, step=epoch)

            t_epoch_start = time.time()
    return G, D
