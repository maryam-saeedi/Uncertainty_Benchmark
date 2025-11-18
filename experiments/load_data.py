import sys
sys.path.append('../Morpho-MNIST')
from datasets.morpho_mnist import MorphoMNISTDataset
from torch.utils.data import DataLoader

def load_data():
    train_dataset = MorphoMNISTDataset('/home/msafa/PhD/morpho/Morpho-MNIST/data/plain/train-images-idx3-ubyte.gz', '/home/msafa/PhD/morpho/Morpho-MNIST/data/plain/train-labels-idx1-ubyte.gz')
    val_dataset = MorphoMNISTDataset('/home/msafa/PhD/morpho/Morpho-MNIST/data/plain/t10k-images-idx3-ubyte.gz', '/home/msafa/PhD/morpho/Morpho-MNIST/data/plain/t10k-labels-idx1-ubyte.gz')
    thining9_dataset = MorphoMNISTDataset('/home/msafa/PhD/morpho/Morpho-MNIST/data/thinning/t10k-images-idx3-ubyte-9.gz', '/home/msafa/PhD/morpho/Morpho-MNIST/data/plain/t10k-labels-idx1-ubyte.gz')
    thining5_dataset = MorphoMNISTDataset('/home/msafa/PhD/morpho/Morpho-MNIST/data/thinning/t10k-images-idx3-ubyte-5.gz', '/home/msafa/PhD/morpho/Morpho-MNIST/data/plain/t10k-labels-idx1-ubyte.gz')

    train_dl = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=1, shuffle=False)
    thining9_dl = DataLoader(thining9_dataset, batch_size=1, shuffle=False)
    thining5_dl = DataLoader(thining5_dataset, batch_size=1, shuffle=False)

    test_dl = {
        'thining 9': thining9_dl,
        'thining 5': thining5_dl,
    }

    return train_dl, val_dl, test_dl