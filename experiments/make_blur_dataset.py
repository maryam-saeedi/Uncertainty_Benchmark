
from datasets.morpho_mnist import MorphoMNISTDataset
from torchvision.transforms.v2 import GaussianBlur
import os

data_path = '/home/msafa/PhD/morpho/Morpho-MNIST/data'

distortions = [GaussianBlur(kernel_size=3, sigma=.5),
               GaussianBlur(kernel_size=5, sigma=1),
               GaussianBlur(kernel_size=11, sigma=2),
               GaussianBlur(kernel_size=17, sigma=3),]

for distortion in distortions:
    ks = distortion.kernel_size
    s = distortion.sigma
    disturbed_train_dataset = MorphoMNISTDataset(os.path.join(data_path, 'plain', 'train-images-idx3-ubyte.gz'),
                                       os.path.join(data_path, 'plain', 'train-labels-idx1-ubyte.gz'),
                                                 transform=distortion)
    disturbed_train_dataset.save_dataset(os.path.join(data_path, 'blured', f'train-images-blur-ks{ks}-s{s}.gz'),
                                         os.path.join(data_path, 'blured', f'train-labels-blur-ks{ks}-s{s}.gz'))
    disturbed_val_dataset = MorphoMNISTDataset(os.path.join(data_path, 'plain', 't10k-images-idx3-ubyte.gz'),
                                     os.path.join(data_path, 'plain', 't10k-labels-idx1-ubyte.gz'),
                                               transform=distortion)
    disturbed_val_dataset.save_dataset(os.path.join(data_path, 'blured', f'val-images-blur-ks{ks}-s{s}.gz'),
                                       os.path.join(data_path, 'blured', f'val-labels-blur-ks{ks}-s{s}.gz'))
