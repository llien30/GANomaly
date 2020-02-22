from torchvision import transforms


# the library for Image Transformation
class MNIST_ImageTransform:
    def __init__(self, CONFIG):
        self.data_transform = transforms.Compose(
            [
                transforms.Resize(CONFIG.input_size, CONFIG.input_size),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def __call__(self, img):
        return self.data_transform(img)
