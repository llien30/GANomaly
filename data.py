import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np


def get_mnist_anomaly_dataset(trn_img, trn_label, tst_img, tst_label, abn_cls):
    nrm_train_idx = torch.from_numpy(np.where(np.array(trn_label) != abn_cls)[0])
    abn_train_idx = torch.from_numpy(np.where(np.array(trn_label) == abn_cls)[0])
    nrm_test_idx = torch.from_numpy(np.where(np.array(tst_label) != abn_cls)[0])
    abn_test_idx = torch.from_numpy(np.where(np.array(tst_label) == abn_cls)[0])

    nrm_train_img = trn_img[nrm_train_idx]  # Normal training images
    abn_train_img = trn_img[abn_train_idx]  # Abnormal training images
    nrm_test_img = tst_img[nrm_test_idx]  # Normal test images
    abn_test_img = tst_img[abn_test_idx]  # Abnormal test images

    nrm_train_label = trn_label[nrm_train_idx]  # Normal training labels
    abn_train_label = trn_label[abn_train_idx]  # Abnormal training labels.
    nrm_test_label = tst_label[nrm_test_idx]  # Normal test labels
    abn_test_label = tst_label[abn_test_idx]  # Abnormal test labels.

    nrm_train_label[:] = 0
    abn_train_label[:] = 1
    nrm_test_label[:] = 0
    abn_test_label[:] = 1

    # Concatenate the downloaded train set and test sets
    nrm_image = torch.cat((nrm_train_img, nrm_test_img), dim=0)
    nrm_label = torch.cat((nrm_train_label, nrm_test_label), dim=0)
    abn_image = torch.cat((abn_train_img, abn_test_img), dim=0)
    abn_label = torch.cat((abn_train_label, abn_test_label), dim=0)

    # split the data
    image_idx = np.arange(len(nrm_label))
    nrm_train_len = int(len(image_idx) * 0.80)
    nrm_train_idx = image_idx[:nrm_train_len]
    nrm_test_idx = image_idx[nrm_train_len:]

    nrm_trn_img = nrm_image[nrm_train_idx]
    nrm_trn_lbl = nrm_label[nrm_train_idx]
    nrm_tst_img = nrm_image[nrm_test_idx]
    nrm_tst_lbl = nrm_label[nrm_test_idx]

    new_trn_img = nrm_trn_img.clone()
    new_trn_lbl = nrm_trn_lbl.clone()
    new_tst_img = torch.cat((nrm_tst_img, abn_image), dim=0)
    new_tst_lbl = torch.cat((nrm_tst_lbl, abn_label), dim=0)

    return new_trn_img, new_trn_lbl, new_tst_img, new_tst_lbl


def get_cifar_anomaly_dataset(trn_img, trn_label, tst_img, tst_label, abn_cls):
    trn_label = np.array(trn_label)
    tst_label = np.array(tst_label)
    nrm_train_idx = np.where(np.array(trn_label) != abn_cls)[0]
    abn_train_idx = np.where(np.array(trn_label) == abn_cls)[0]
    nrm_test_idx = np.where(np.array(tst_label) != abn_cls)[0]
    abn_test_idx = np.where(np.array(tst_label) == abn_cls)[0]

    nrm_train_img = trn_img[nrm_train_idx]  # Normal training images
    abn_train_img = trn_img[abn_train_idx]  # Abnormal training images
    nrm_test_img = tst_img[nrm_test_idx]  # Normal test images
    abn_test_img = tst_img[abn_test_idx]  # Abnormal test images

    nrm_train_label = trn_label[nrm_train_idx]  # Normal training labels
    abn_train_label = trn_label[abn_train_idx]  # Abnormal training labels.
    nrm_test_label = tst_label[nrm_test_idx]  # Normal test labels
    abn_test_label = tst_label[abn_test_idx]  # Abnormal test labels.

    nrm_train_label[:] = 0
    abn_train_label[:] = 1
    nrm_test_label[:] = 0
    abn_test_label[:] = 1

    # Concatenate the downloaded train set and test sets
    nrm_image = np.concatenate((nrm_train_img, nrm_test_img), axis=0)
    nrm_label = np.concatenate((nrm_train_label, nrm_test_label), axis=0)
    abn_image = np.concatenate((abn_train_img, abn_test_img), axis=0)
    abn_label = np.concatenate((abn_train_label, abn_test_label), axis=0)

    # split the data
    image_idx = np.arange(len(nrm_label))
    nrm_train_len = int(len(image_idx) * 0.80)
    nrm_train_idx = image_idx[:nrm_train_len]
    nrm_test_idx = image_idx[nrm_train_len:]

    nrm_trn_img = nrm_image[nrm_train_idx]
    nrm_trn_lbl = nrm_label[nrm_train_idx]
    nrm_tst_img = nrm_image[nrm_test_idx]
    nrm_tst_lbl = nrm_label[nrm_test_idx]

    new_trn_img = np.copy(nrm_trn_img)
    new_trn_lbl = np.copy(nrm_trn_lbl)
    new_tst_img = np.concatenate((nrm_tst_img, abn_image), axis=0)
    new_tst_lbl = np.concatenate((nrm_tst_lbl, abn_label), axis=0)

    return new_trn_img, new_trn_lbl, new_tst_img, new_tst_lbl


def load_data(CONFIG):

    if CONFIG.dataset == "MNIST":
        transform = transforms.Compose(
            [
                transforms.Resize((CONFIG.input_size, CONFIG.input_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        train_dataset = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )

        (
            train_dataset.data,
            train_dataset.targets,
            test_dataset.data,
            test_dataset.targets,
        ) = get_mnist_anomaly_dataset(
            trn_img=train_dataset.data,
            trn_label=train_dataset.targets,
            tst_img=test_dataset.data,
            tst_label=test_dataset.targets,
            abn_cls=CONFIG.abnormal_class,
        )

        dataloader = {
            "train": DataLoader(
                dataset=train_dataset, batch_size=CONFIG.batch_size, shuffle=True,
            ),
            "test": DataLoader(
                dataset=test_dataset, batch_size=CONFIG.test_batch_size, shuffle=True,
            ),
        }

        return dataloader

    if CONFIG.dataset == "CIFAR10":
        classes = {
            "plane": 0,
            "car": 1,
            "bird": 2,
            "cat": 3,
            "deer": 4,
            "dog": 5,
            "frog": 6,
            "horse": 7,
            "ship": 8,
            "truck": 9,
        }
        transform = transforms.Compose(
            [
                transforms.Resize((CONFIG.input_size, CONFIG.input_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        train_dataset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )

        (
            train_dataset.data,
            train_dataset.targets,
            test_dataset.data,
            test_dataset.targets,
        ) = get_cifar_anomaly_dataset(
            trn_img=train_dataset.data,
            trn_label=train_dataset.targets,
            tst_img=test_dataset.data,
            tst_label=test_dataset.targets,
            abn_cls=classes[CONFIG.abnormal_class],
        )

        dataloader = {
            "train": DataLoader(
                dataset=train_dataset, batch_size=CONFIG.batch_size, shuffle=True,
            ),
            "test": DataLoader(
                dataset=test_dataset, batch_size=CONFIG.test_batch_size, shuffle=True,
            ),
        }

        return dataloader
