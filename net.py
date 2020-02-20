import torch.nn as nn


def make_Encoder(
    input_size,
    z_dim,
    channel,
    ndf,
    n_extra_layers=0,
    add_final_conv=True,
    discrimination=False,
):
    """
    Encoder Network(like DCGAN Discriminator network)

    input_size : the image size of the data
    z_dim : the dimention of the latent space
    channel : the number of channels of the data
    ndf : the number of filters in encoder(same as decoder)
    """
    assert input_size % 16 == 0, "input_size has to be a multiple of 16"

    main = nn.Sequential()
    # input is channel x input_size x input_size
    main.add_module(
        "initial_conv-{}-{}".format(channel, ndf),
        nn.Conv2d(channel, ndf, kernel_size=4, stride=2, padding=1, bias=False),
    )
    main.add_module("initial_ReLU-{}".format(ndf), nn.LeakyReLU(0.2, inplace=True))
    csize, cndf = input_size / 2, ndf

    # Extra layers does not change channel size
    for t in range(n_extra_layers):
        main.add_module(
            "extra_conv-{}-{}".format(t, cndf),
            nn.Conv2d(cndf, cndf, kernel_size=3, stride=1, padding=1, bias=False),
        )
        main.add_module("extra_BatchNorm-{}-{}".format(t, cndf), nn.BatchNorm2d(cndf))
        main.add_module(
            "extra_LeakyReLU-{}-{}".format(t, cndf), nn.LeakyReLU(0.2, inplace=True)
        )

    # cndf is customed to the size of the input
    # csize : size of the convolution
    while csize > 4:
        in_feat = cndf  # the number of input feature
        out_feat = cndf * 2  # the number of output feature
        main.add_module(
            "pyramid_conv-{}-{}".format(in_feat, out_feat),
            nn.Conv2d(
                in_feat, out_feat, kernel_size=4, stride=2, padding=1, bias=False
            ),
        )
        main.add_module(
            "pyramid_BatchNorm-{}".format(out_feat), nn.BatchNorm2d(out_feat)
        )
        main.add_module(
            "pyramid_LeakyReLU-{}".format(out_feat), nn.LeakyReLU(0.2, inplace=True)
        )
        cndf = cndf * 2
        csize = csize // 2

    if add_final_conv:
        main.add_module(
            "final-{}-{}".format(cndf, z_dim),
            nn.Conv2d(cndf, z_dim, kernel_size=4, stride=1, padding=0, bias=False),
        )
        # the output image size is 1

    if discrimination:
        main.add_module("Linear", nn.Linear(z_dim, 1))
    return main


def make_Decoder(input_size, z_dim, channel, ngf, n_extralayers=0):
    """
    DCGAN Decoder Network
    (The symmetry of Encoder)
    """
    assert input_size % 16 == 0, "input_size has to be a multiple of 16"

    main = nn.Sequential()
    # upsize cngf untill the size that the last layer's cngf is equals to ngf
    cngf, tisize = ngf // 2, 4
    while tisize != input_size:
        cngf = cngf * 2
        tisize = tisize * 2

    # Decoder's input is z
    main.add_module(
        "initial_convt-{}-{}".format(z_dim, cngf),
        nn.ConvTranspose2d(z_dim, cngf, kernel_size=4, stride=1, padding=0, bias=False),
    )
    # output size = 4
    main.add_module("initial_BatchNorm-{}".format(cngf), nn.BatchNorm2d(cngf))
    main.add_module("initial_ReLU-{}".format(cngf), nn.ReLU(inplace=True))

    csize = 4
    while csize < input_size // 2:
        main.add_module(
            "pyramid_convt-{}-{}".format(cngf, cngf // 2),
            nn.ConvTranspose2d(
                cngf, cngf // 2, kernel_size=4, stride=2, padding=1, bias=False
            ),
        )
        main.add_module(
            "pyramid_BatchNorm-{}".format(cngf // 2), nn.BatchNorm2d(cngf // 2)
        )
        main.add_module("pyramid_ReLU-{}".format(cngf // 2), nn.ReLU(inplace=True))

        csize = csize * 2
        cngf = cngf // 2

    if cngf != ngf:
        print("ngf is wrong size")

    # extra layers don't change both channel size and image size
    for t in range(n_extralayers):
        main.add_module(
            "extra_convt-{}-{}".format(cngf, cngf),
            nn.ConvTranspose2d(
                cngf, cngf, kernel_size=3, stride=1, padding=1, bias=False
            ),
        )
        main.add_module("extra_BatchNorm-{}".format(cngf), nn.BatchNorm2d(cngf))
        main.add_module("extra_ReLU-{}".format(cngf), nn.ReLU(inplace=True))

    main.add_module(
        "final_convt-{}-{}".format(ngf, channel),
        nn.ConvTranspose2d(
            ngf, channel, kernel_size=4, stride=2, padding=1, bias=False
        ),
    )
    main.add_module("final_tanh-{}".format(channel), nn.Tanh())

    return main


class NetD(nn.Module):
    """
    Discriminator Network
    """

    def __init__(self, CONFIG):
        super(NetD, self).__init__()

        model = make_Encoder(
            CONFIG.input_size,
            CONFIG.z_dim,
            CONFIG.channel,
            CONFIG.ndf,
            CONFIG.extralayers,
            True,
            True,
        )
        layers = list(model.children())

        # to output feature, separete the network
        self.feature = nn.Sequential(*layers[:-2])
        self.classifier = nn.Sequential(layers[-2])
        self.discriminator = nn.Sequential(layers[-1])
        # self.classifier.add_module("Sigmoid", nn.Sigmoid())

    def forward(self, x, CONFIG):
        feature = self.feature(x)
        classifier = self.classifier(feature)
        # output the classification as a vector
        classifier = classifier.reshape(-1, CONFIG.z_dim)
        classifier = self.discriminator(classifier)
        classifier = classifier.view(-1)

        return classifier, feature


class NetG(nn.Module):
    """
    Generator Network
    """

    def __init__(self, CONFIG):
        super(NetG, self).__init__()
        self.encoder1 = make_Encoder(
            CONFIG.input_size,
            CONFIG.z_dim,
            CONFIG.channel,
            CONFIG.ndf,
            CONFIG.extralayers,
        )
        self.decoder = make_Decoder(
            CONFIG.input_size,
            CONFIG.z_dim,
            CONFIG.channel,
            CONFIG.ngf,
            CONFIG.extralayers,
        )
        self.encoder2 = make_Encoder(
            CONFIG.input_size,
            CONFIG.z_dim,
            CONFIG.channel,
            CONFIG.ndf,
            CONFIG.extralayers,
        )

    def forward(self, x):
        latent_i = self.encoder1(x)
        gan_img = self.decoder(latent_i)
        latent_o = self.encoder2(gan_img)

        return gan_img, latent_i, latent_o
