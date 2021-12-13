import torchvision.transforms as transforms
from torch.utils.data import Dataset, WeightedRandomSampler
import torch
import numpy as np

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 64 x 64 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set


def data_transforms(input_size=128, augment=False):
    ops = [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    if augment:
        # small augment
        # ops.append(transforms.RandomAffine(degrees=(-20, 20), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10)))
        # medium augment
        # ops.append(transforms.RandomAffine(degrees=(-40, 40), translate=(0.3, 0.3), scale=(0.7, 1.3), shear=(-30, 30)))
        # big augment
        ops.append(transforms.RandomAffine(degrees=(-80, 80), translate=(0.5, 0.5), scale=(0.5, 1.5), shear=(-40, 40)))
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
    return transforms.Compose(ops)


class Subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        im, labels = self.dataset[self.indices[idx]]
        return self.transform(im), labels

    def __len__(self):
        return len(self.indices)


def get_weighted_sampler(dataset):
    targets = np.array([l for (_, l) in dataset])

    hard_weights = np.array([0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 4, 4, 4, 0]) + 1.
    samples_weight = hard_weights[targets]
    samples_weight = torch.from_numpy(samples_weight / samples_weight.sum())
    return WeightedRandomSampler(samples_weight, len(samples_weight))
