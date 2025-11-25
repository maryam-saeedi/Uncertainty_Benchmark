import sys
from distutils.command.config import config

sys.path.append('../Morpho-MNIST')
from datasets.morpho_mnist import MorphoMNISTDataset
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
import os
import matplotlib.pyplot as plt

def show_subsset(dl):
    single_batch = next((iter(dl)))
    fig = plt.figure(figsize=(16, 6))
    for i in range(16):
        ax = fig.add_subplot(4, 8, i+1, xticks=[], yticks=[])
        image, label = single_batch[0][i].numpy().squeeze(), single_batch[1][i].numpy().squeeze()
        ax.imshow(image, cmap='gray')
        ax.set_title(label, fontsize=15, color='green')
    plt.tight_layout()
    plt.show()

def load_data():
    train_dataset = MorphoMNISTDataset('/home/msafa/PhD/morpho/Morpho-MNIST/data/plain/train-images-idx3-ubyte.gz', '/home/msafa/PhD/morpho/Morpho-MNIST/data/plain/train-labels-idx1-ubyte.gz')
    val_dataset = MorphoMNISTDataset('/home/msafa/PhD/morpho/Morpho-MNIST/data/plain/t10k-images-idx3-ubyte.gz', '/home/msafa/PhD/morpho/Morpho-MNIST/data/plain/t10k-labels-idx1-ubyte.gz')
    thining9_dataset = MorphoMNISTDataset('/home/msafa/PhD/morpho/Morpho-MNIST/data/thinning/t10k-images-idx3-ubyte-9.gz', '/home/msafa/PhD/morpho/Morpho-MNIST/data/plain/t10k-labels-idx1-ubyte.gz')
    thining5_dataset = MorphoMNISTDataset('/home/msafa/PhD/morpho/Morpho-MNIST/data/thinning/t10k-images-idx3-ubyte-5.gz', '/home/msafa/PhD/morpho/Morpho-MNIST/data/plain/t10k-labels-idx1-ubyte.gz')
    thining3_dataset = MorphoMNISTDataset('/home/msafa/PhD/morpho/Morpho-MNIST/data/thinning/t10k-images-idx3-ubyte-3.gz', '/home/msafa/PhD/morpho/Morpho-MNIST/data/plain/t10k-labels-idx1-ubyte.gz')
    fracture10_dataset = MorphoMNISTDataset('/home/msafa/PhD/morpho/Morpho-MNIST/data/fracture/t10k-images-idx3-ubyte-10.gz', '/home/msafa/PhD/morpho/Morpho-MNIST/data/plain/t10k-labels-idx1-ubyte.gz')
    fracture5_dataset = MorphoMNISTDataset(
        '/home/msafa/PhD/morpho/Morpho-MNIST/data/fracture/t10k-images-idx3-ubyte-5.gz',
        '/home/msafa/PhD/morpho/Morpho-MNIST/data/plain/t10k-labels-idx1-ubyte.gz')
    fracture3_dataset = MorphoMNISTDataset(
        '/home/msafa/PhD/morpho/Morpho-MNIST/data/fracture/t10k-images-idx3-ubyte-3.gz',
        '/home/msafa/PhD/morpho/Morpho-MNIST/data/plain/t10k-labels-idx1-ubyte.gz')

    train_dl = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=1, shuffle=False)
    thining9_dl = DataLoader(thining9_dataset, batch_size=1, shuffle=False)
    thining5_dl = DataLoader(thining5_dataset, batch_size=1, shuffle=False)
    thining3_dl = DataLoader(thining3_dataset, batch_size=1, shuffle=False)
    fracture10_dl = DataLoader(fracture10_dataset, batch_size=1, shuffle=False)
    fracture5_dl = DataLoader(fracture5_dataset, batch_size=1, shuffle=False)
    fracture3_dl = DataLoader(fracture3_dataset, batch_size=1, shuffle=False)

    test_dl = {
        'thining 9': thining9_dl,
        'thining 5': thining5_dl,
        'thining 3': thining3_dl,
        # 'fracture 10': fracture10_dl,
        # 'fracture 5': fracture5_dl,
        # 'fracture 3': fracture3_dl,
        # 'fracture 3': fracture3_dl,
    }

    return train_dl, val_dl, test_dl

def load_aleatoric_data(data_path):
    train_datasets = {}
    val_datasets = {}
    plain_train_dataset = MorphoMNISTDataset(os.path.join(data_path, 'plain', 'train-images-idx3-ubyte.gz'),
                                       os.path.join(data_path, 'plain', 'train-labels-idx1-ubyte.gz'))
    plain_val_dataset = MorphoMNISTDataset(os.path.join(data_path, 'plain', 't10k-images-idx3-ubyte.gz'),
                                     os.path.join(data_path, 'plain', 't10k-labels-idx1-ubyte.gz'))
    blured_train_dataset = MorphoMNISTDataset(os.path.join(data_path, 'blured', 'train-images-blur-ks(17, 17)-s[3.0, 3.0].gz'),
                                              os.path.join(data_path, 'blured', 'train-labels-blur-ks(17, 17)-s[3.0, 3.0].gz'),)
    blured_val_dataset = MorphoMNISTDataset(os.path.join(data_path, 'blured', 'val-images-blur-ks(17, 17)-s[3.0, 3.0].gz'),
                                            os.path.join(data_path, 'blured', 'val-labels-blur-ks(17, 17)-s[3.0, 3.0].gz'))
    train_datasets['high_sever'] = ConcatDataset([plain_train_dataset, blured_train_dataset])
    val_datasets['high_sever'] = ConcatDataset([plain_val_dataset, blured_val_dataset])

    blured_train_dataset = MorphoMNISTDataset(
        os.path.join(data_path, 'blured', 'train-images-blur-ks(11, 11)-s[2.0, 2.0].gz'),
        os.path.join(data_path, 'blured', 'train-labels-blur-ks(11, 11)-s[2.0, 2.0].gz'), )
    blured_val_dataset = MorphoMNISTDataset(
        os.path.join(data_path, 'blured', 'val-images-blur-ks(11, 11)-s[2.0, 2.0].gz'),
        os.path.join(data_path, 'blured', 'val-labels-blur-ks(11, 11)-s[2.0, 2.0].gz'))
    train_datasets['med_sever'] = ConcatDataset([plain_train_dataset, blured_train_dataset])
    val_datasets['med_sever'] = ConcatDataset([plain_val_dataset, blured_val_dataset])


    blured_train_dataset = MorphoMNISTDataset(
        os.path.join(data_path, 'blured', 'train-images-blur-ks(5, 5)-s[1.0, 1.0].gz'),
        os.path.join(data_path, 'blured', 'train-labels-blur-ks(5, 5)-s[1.0, 1.0].gz'), )
    blured_val_dataset = MorphoMNISTDataset(
        os.path.join(data_path, 'blured', 'val-images-blur-ks(5, 5)-s[1.0, 1.0].gz'),
        os.path.join(data_path, 'blured', 'val-labels-blur-ks(5, 5)-s[1.0, 1.0].gz'))
    train_datasets['low_sever'] = ConcatDataset([plain_train_dataset, blured_train_dataset])
    val_datasets['low_sever'] = ConcatDataset([plain_val_dataset, blured_val_dataset])

    train_loaders = {}
    val_loaders = {}
    for name in train_datasets.keys():
        train_loaders[name] = DataLoader(train_datasets[name], batch_size=64, shuffle=True)
        val_loaders[name] = DataLoader(val_datasets[name], batch_size=1, shuffle=False)

    return train_loaders, val_loaders