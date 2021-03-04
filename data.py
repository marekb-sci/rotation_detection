# -*- coding: utf-8 -*-

import torchvision
import numpy as np

# %%
class MyTransform:
    def __init__(self, transform, deg_range=90, radian_out=False):
        self.transform = transform
        self.deg_range = deg_range
        self.radian_out = radian_out
    def __call__(self, inputs, target):
        deg = (np.random.rand()-0.5)*self.deg_range

        deg_out = np.radians(deg) if self.radian_out else deg

        return self.transform(inputs.rotate(deg)), np.float32(deg_out)

# %%
def get_dataset(deg_range=60, scale=512/720, crop=224, radian_out=False, random_crop=True):
    tr = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(720),
        torchvision.transforms.Resize(int(720*scale)),
        torchvision.transforms.RandomCrop(crop) if random_crop else torchvision.transforms.CenterCrop(crop),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        ])

    return torchvision.datasets.Cityscapes(r'D:\datasets\city', transforms=MyTransform(tr, deg_range, radian_out=radian_out))


# %%
if __name__ == '__main__':

    dataset = torchvision.datasets.Cityscapes(r'D:\datasets\city')

    #%%
    def rotate(pil_image, degrees):
        return pil_image.rotate(degrees)

    def rotate45(pil_image):
        return pil_image.rotate(45)

    tr = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(rotate45),
        torchvision.transforms.CenterCrop(720),
        torchvision.transforms.Resize(512),
        torchvision.transforms.RandomCrop(224)
        ])

    images = [tr(dataset[i][0]) for i in range(20)]

    #%%
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(4,5)
    for i, img in enumerate(images):
        axs[i%4][i//4].imshow(img)

    # %%
    tr = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(720),
        torchvision.transforms.Resize(512),
        torchvision.transforms.RandomCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        ])

    dataset = torchvision.datasets.Cityscapes(r'D:\datasets\city', transforms=MyTransform(tr))

    img0, deg0 = dataset[0]

    # %%
    import torch
    model = torchvision.models.densenet121(pretrained=True)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 1)

    dataloader = torch.utils.data.DataLoader(dataset)
    batch0 = next(iter(dataloader))
    out = model(batch0[0])