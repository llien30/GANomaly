import torch

from libs.roc import roc

# import time


def test(CONFIG, epoch, G, test_dataloader, device):
    # where to save scores
    Scores = torch.zeros(
        size=(len(test_dataloader.dataset),), dtype=torch.float32, device="cpu",
    )
    # where to save test labels
    Labels = torch.zeros(
        size=(len(test_dataloader.dataset),), dtype=torch.long, device="cpu"
    )
    mini_batch_size = CONFIG.test_batch_size
    total_steps = 0

    # times = []
    for i, (imges, label) in enumerate(test_dataloader, 0):
        mini_batch_size = imges.size()[0]
        total_steps = total_steps + mini_batch_size
        # start_time = time.time()

        imges = imges.reshape(-1, CONFIG.channel, CONFIG.input_size, CONFIG.input_size)
        imges = imges.to(device)
        G.eval()
        with torch.no_grad():
            _, latent_i, latent_o = G(imges)
        # caliculate the img's error
        error = torch.mean(torch.pow((latent_i - latent_o), 2), dim=1).to("cpu")

        # last_time = time.time()
        Scores[
            i * mini_batch_size : i * mini_batch_size + error.size(0)
        ] = error.reshape(error.size(0))
        Labels[
            i * mini_batch_size : i * mini_batch_size + error.size(0)
        ] = label.reshape(error.size(0))

        # times.append(last_time - start_time)
    # Normalize score
    Scores = (Scores - torch.min(Scores)) / (torch.max(Scores) - torch.min(Scores))
    Acc = roc(Labels, Scores)

    print("Epoch {}/{} || Accuracy(ROC):{}".format(epoch, CONFIG.num_epochs, Acc))
    return Acc
