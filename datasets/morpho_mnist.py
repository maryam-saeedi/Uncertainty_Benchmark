import gzip
import torch
from torch.utils.data import Dataset
import struct
import numpy as np
import sys, os
from multipledispatch import dispatch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Morpho-MNIST')))
from morphomnist import morpho, perturb


def pars_gzip_file(path):
    '''
    a function to parse a gzipped file and return its data in numpy array format.
    :param path: path to gzipped file
    :return: ndarray data
    '''
    with gzip.open(path, 'rb') as f:
        idx_dtype, ndim = struct.unpack('BBBB', f.read(4))[2:]
        shape = struct.unpack('>' + 'I' * ndim, f.read(4 * ndim))
        buffer_length = int(np.prod(shape))
        data = np.frombuffer(f.read(buffer_length), dtype=np.uint8).reshape(shape)
    return data


class UseMorpho(object):
    def __init__(self, thinning, thickening, swelling, fractures):
        self.perturbations = (
            perturb.Thinning(amount=thinning),
            perturb.Thickening(amount=thickening),
            perturb.Swelling(strength=swelling[0], radius=swelling[1]),
            perturb.Fracture(num_frac=fractures),
        )

    def __call__(self, image):
        morphology = morpho.ImageMorphology(image, scale=4)
        perturbation = self.perturbations[np.random.randint(len(self.perturbations))]
        perturbed_image = perturbation(morphology)
        perturbed_image = morphology.downscale(perturbed_image)
        return perturbed_image

    def __repr__(self):
        return f"{self.__class__.__name__}"

class MorphoMNISTDataset(Dataset):
    '''
    Custom dataset class for Morpho-MNIST dataset.
    data can be read from gzip file (as provided by morpho-mnist repo to download) or ndarray data.
    **both images and labels should be in the same format.
    morpho-mnist perturbation can be applied to the data while fetching and returning them.
    '''

    @dispatch(str, str)
    def __init__(self, images, labels, perturbation=None, transform=None):
        if (isinstance(images, str) and images.endswith(".gz")) and (isinstance(labels, str) and labels.endswith(".gz")):   ## if the path to the gzip file of data is provided
            self.images = pars_gzip_file(images)
            self.labels = pars_gzip_file(labels)

        self.transform = transform
        self.perturbation = perturbation    ## should be one of morpho-mnist perturbation function

    @dispatch(np.ndarray, np.ndarray)
    def __init__(self, images, labels, perturbation=None, transform=None):   ## if raw data is in numpy format
        self.images = images
        self.labels = labels

        self.transform = transform
        self.perturbation = perturbation    ## should be one of morpho-mnist perturbation function


    @dispatch(torch.utils.data.Dataset)
    def __init__(self, dataset, perturbation=None, transform=None):
        self.images, self.labels = [], []
        for item in dataset:
            self.images.append(item[0])
            self.labels.append(item[1])
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)

        self.transform = transform
        self.perturbation = perturbation

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.perturbation is not None:       ## perturb image based on the perturbation function, this part is used morpho-mnist structure
            image = self.perturbation(image)
        if self.transform is not None:
            image = self.transform(image)
        return torch.tensor(image).float(), torch.tensor(label)     ## return image (in raw or perturbed version) and corresponding label, both in tensor format