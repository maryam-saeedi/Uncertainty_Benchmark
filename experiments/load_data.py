import sys

import numpy as np
from distutils.command.config import config
from skimage.graph import rag_mean_color
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sympy.printing.llvmjitcode import link_names
from torchvision import transforms
import hydra
import torch
from make_noisy_label import *

sys.path.append('../Morpho-MNIST')

from morphomnist import morpho, perturb
from datasets.morpho_mnist import MorphoMNISTDataset, pars_gzip_file
from datasets.isic import SkinISICDataset
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
import os
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.transforms.v2 import GaussianBlur, RandomApply

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

def load_data(data_path):
    # normalize = transforms.Normalize((0.1307,), (0.3081,))
    normalize = None
    train_dataset = MorphoMNISTDataset(os.path.join(data_path,'plain', 'train-images-idx3-ubyte.gz'), os.path.join(data_path,'plain', 'train-labels-idx1-ubyte.gz'), transform=normalize)
    val_dataset = MorphoMNISTDataset(os.path.join(data_path,'plain', 't10k-images-idx3-ubyte.gz'), os.path.join(data_path,'plain', 't10k-labels-idx1-ubyte.gz'), transform=normalize)
    thining9_dataset = MorphoMNISTDataset(os.path.join(data_path,'thinning', 't10k-images-idx3-ubyte-9.gz'), os.path.join(data_path,'plain', 't10k-labels-idx1-ubyte.gz'), transform=normalize)
    thining7_dataset = MorphoMNISTDataset(os.path.join(data_path,'thinning', 't10k-images-idx3-ubyte-7.gz'), os.path.join(data_path,'plain', 't10k-labels-idx1-ubyte.gz'), transform=normalize)
    thining5_dataset = MorphoMNISTDataset(os.path.join(data_path,'thinning', 't10k-images-idx3-ubyte-5.gz'), os.path.join(data_path,'plain', 't10k-labels-idx1-ubyte.gz'), transform=normalize)
    thining3_dataset = MorphoMNISTDataset(os.path.join(data_path,'thinning', 't10k-images-idx3-ubyte-3.gz'), os.path.join(data_path,'plain', 't10k-labels-idx1-ubyte.gz'), transform=normalize)
    thining1_dataset = MorphoMNISTDataset(os.path.join(data_path,'thinning', 't10k-images-idx3-ubyte-1.gz'), os.path.join(data_path,'plain', 't10k-labels-idx1-ubyte.gz'), transform=normalize)
    fracture10_dataset = MorphoMNISTDataset(os.path.join(data_path,'fracture', 't10k-images-idx3-ubyte-10.gz'), os.path.join(data_path,'plain', 't10k-labels-idx1-ubyte.gz'), transform=normalize)
    fracture5_dataset = MorphoMNISTDataset(
        os.path.join(data_path,'fracture', 't10k-images-idx3-ubyte-5.gz'),
        os.path.join(data_path,'plain', 't10k-labels-idx1-ubyte.gz'), transform=normalize)
    fracture3_dataset = MorphoMNISTDataset(
        os.path.join(data_path,'fracture', 't10k-images-idx3-ubyte-3.gz'),
        os.path.join(data_path,'plain', 't10k-labels-idx1-ubyte.gz'), transform=normalize)
    fracture1_dataset = MorphoMNISTDataset(
        os.path.join(data_path,'fracture', 't10k-images-idx3-ubyte-1.gz'),
        os.path.join(data_path,'plain', 't10k-labels-idx1-ubyte.gz'), transform=normalize)

    train_dl = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=1, shuffle=False)
    thining9_dl = DataLoader(thining9_dataset, batch_size=1, shuffle=False)
    thining7_dl = DataLoader(thining7_dataset, batch_size=1, shuffle=False)
    thining5_dl = DataLoader(thining5_dataset, batch_size=1, shuffle=False)
    thining3_dl = DataLoader(thining3_dataset, batch_size=1, shuffle=False)
    thining1_dl = DataLoader(thining1_dataset, batch_size=1, shuffle=False)
    fracture10_dl = DataLoader(fracture10_dataset, batch_size=1, shuffle=False)
    fracture5_dl = DataLoader(fracture5_dataset, batch_size=1, shuffle=False)
    fracture3_dl = DataLoader(fracture3_dataset, batch_size=1, shuffle=False)
    fracture1_dl = DataLoader(fracture1_dataset, batch_size=1, shuffle=False)

    test_dl = {
        'thining': {
        'thining 1': thining1_dl,
        'thining 3': thining3_dl,
        'thining 5': thining5_dl,
        'thining 7': thining7_dl,
        'thining 9': thining9_dl,},
        'fracture': {
        'fracture 1': fracture1_dl,
        'fracture 3': fracture3_dl,
        'fracture 5': fracture5_dl,
        'fracture 10': fracture10_dl,}
    }

    return train_dl, val_dl, test_dl

def load_data_2(data_path):

    perturbations = {
        "thinning": {'thinning 1': perturb.Thinning(amount=.1),
                     'thinning 3': perturb.Thinning(amount=.3),
                     'thinning 7': perturb.Thinning(amount=.7),
                     'thinning 9': perturb.Thinning(amount=.9)
                     },
        "thickening": {
            "thickening 50": perturb.Thickening(amount=.5),
            "thickening 100": perturb.Thickening(amount=1),
            "thickening 200": perturb.Thickening(amount=2),
            "thickening 300": perturb.Thickening(amount=3)
        }
    }

    train_dataset = MorphoMNISTDataset(os.path.join(data_path, 'plain', 'train-images-idx3-ubyte.gz'),
                                       os.path.join(data_path, 'plain', 'train-labels-idx1-ubyte.gz'))
    val_dataset = MorphoMNISTDataset(os.path.join(data_path, 'plain', 't10k-images-idx3-ubyte.gz'),
                                     os.path.join(data_path, 'plain', 't10k-labels-idx1-ubyte.gz'))

    train_dl = DataLoader(train_dataset, batch_size=128, shuffle=False)
    val_dl = DataLoader(val_dataset, batch_size=16, shuffle=False)

    test_dl = {}
    for name in perturbations.keys():
        test_dl[name] = {}
        for per in perturbations[name].keys():
            test_dataset = MorphoMNISTDataset(os.path.join(data_path, 'plain', 't10k-images-idx3-ubyte.gz'),
                                     os.path.join(data_path, 'plain', 't10k-labels-idx1-ubyte.gz'), perturbation=perturbations[name][per])
            test_dl[name][per] = DataLoader(test_dataset, batch_size=16, shuffle=False)

    return train_dl, val_dl, test_dl


def load_mnist_blur_data(data_path):
    train_loaders = {}
    val_loaders = {}
    for p in np.arange(0.0, 1.1, 0.1):
        distortion = RandomApply([GaussianBlur(kernel_size=5, sigma=1)], p)
        disturbed_train_dataset = MorphoMNISTDataset(os.path.join(data_path, 'plain', 'train-images-idx3-ubyte.gz'),
                                                     os.path.join(data_path, 'plain', 'train-labels-idx1-ubyte.gz'),
                                                     transform=distortion)
        disturbed_val_dataset = MorphoMNISTDataset(os.path.join(data_path, 'plain', 't10k-images-idx3-ubyte.gz'),
                                                   os.path.join(data_path, 'plain', 't10k-labels-idx1-ubyte.gz'),
                                                   transform=distortion)
        name = int(p*100)
        train_loaders[name] = DataLoader(disturbed_train_dataset, batch_size=64, shuffle=True)
        val_loaders[name] = DataLoader(disturbed_val_dataset, batch_size=1, shuffle=False)

    return train_loaders, val_loaders

def load_mnist_noisy_label(cfg):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MLP(cfg)
    model.to(device)
    model.load_state_dict(
        torch.load('/home/msafa/PhD/benchmark/experiments/morpho_mnist_result/common/base_model_mlp.pt',
                   weights_only=True, map_location=device))
    print(model)

    train_loaders = {}
    val_loaders = {}

    for ratio in [0.1, 0.2, 0.5]:
        dataset = MorphoMNISTDataset('/home/msafa/PhD/morpho/Morpho-MNIST/data/plain/train-images-idx3-ubyte.gz',
                                     '/home/msafa/PhD/morpho/Morpho-MNIST/data/plain/train-labels-idx1-ubyte.gz')
        print(len(dataset))
        embeddings, labels = extract_embeddings(model, dataset, device=device)

        new_labels, clean_labels, noisy_ids = inject_structured_noise_centroid_flip(
            embeddings,
            labels,
            noise_ratio=ratio,
            neighbors_k=15,
            per_class_anchor_ratio=0.05
        )

        dataset.targets = new_labels
        train, test = train_test_split(dataset, test_size=0.2)
        train_loaders[ratio] = DataLoader(train, batch_size=64, shuffle=True)
        val_loaders[ratio] = DataLoader(test, batch_size=16, shuffle=False)

    return train_loaders, val_loaders

def load_isic_data(data_path):
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),

        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),

        transforms.RandomApply([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            ),
            # v2.RandomChoice([v2.CutMix(num_classes=6), v2.MixUp(num_classes=6)])

        ], p=0.7),  # 70% chance to apply ONE OR MORE from the group

        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        ),
    ])

    metadata = pd.read_csv(os.path.join(data_path, 'MILK10k_Training_Metadata.csv'))
    data = pd.read_csv(os.path.join(data_path, 'MILK10k_Training_GroundTruth.csv'))
    # %%
    data = pd.merge(data, metadata, how='left', on='lesion_id')
    data = data[data['image_type'] == 'dermoscopic']
    # %%
    classes = ['AKIEC', 'BCC', 'BEN_OTH', 'BKL', 'DF', 'INF', 'MAL_OTH', 'MEL', 'NV', 'SCCKA', 'VASC']
    data['label'] = data[classes].idxmax(axis=1)
    classes = ['BCC', 'SCCKA', 'AKIEC', 'NV', 'BKL', 'MEL']
    data = data[data['label'].isin(classes)]
    data['label'] = data[classes].values.argmax(axis=1)
    data['image_id'] = data.apply(lambda x : x['lesion_id']+'/'+x['isic_id'], axis=1)

    # train_df = pd.read_csv(os.path.join(data_path, 'train_6cls.csv'))
    # val_df = pd.read_csv(os.path.join(data_path, 'val_6cls.csv'))

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_ids, val_ids) in enumerate(kfold.split(data, data['label'])):
        train_df = data.iloc[train_ids]  # train set
        val_df = data.iloc[val_ids]  # validation set

        train_dataset = SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), train_df, transform=train_transform)
        val_dataset = SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), val_df, transform=val_transform)

        train_dl = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_dl = DataLoader(val_dataset, batch_size=1, shuffle=False)

        test_dl = {
            'age': {
                'under 30': DataLoader(SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), val_df[val_df['age_approx']<=30], transform=val_transform), batch_size=1, shuffle=False),
                '35': DataLoader(SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), val_df[val_df['age_approx']==35], transform=val_transform), batch_size=1, shuffle=False),
                '40': DataLoader(SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), val_df[val_df['age_approx']==40], transform=val_transform), batch_size=1, shuffle=False),
                '45': DataLoader(SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), val_df[val_df['age_approx']==45], transform=val_transform), batch_size=32, shuffle=False),
                '50': DataLoader(SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), val_df[val_df['age_approx']==50], transform=val_transform), batch_size=32, shuffle=False),
                '55': DataLoader(SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), val_df[val_df['age_approx']==55], transform=val_transform), batch_size=32, shuffle=False),
                '60': DataLoader(SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), val_df[val_df['age_approx']==60], transform=val_transform), batch_size=32, shuffle=False),
                '65': DataLoader(SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), val_df[val_df['age_approx']==65], transform=val_transform), batch_size=32, shuffle=False),
                '70': DataLoader(SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), val_df[val_df['age_approx']==70], transform=val_transform), batch_size=32, shuffle=False),
                '75': DataLoader(SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), val_df[val_df['age_approx']==75], transform=val_transform), batch_size=32, shuffle=False),
                '80': DataLoader(SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), val_df[val_df['age_approx']==80], transform=val_transform), batch_size=32, shuffle=False),
                '85': DataLoader(SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), val_df[val_df['age_approx']==85], transform=val_transform), batch_size=32, shuffle=False),
            },
            'skin tone': {
                'tone 1': DataLoader(SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), val_df[val_df['skin_tone_class']==1], transform=val_transform), batch_size=32, shuffle=False),
                'tone 2': DataLoader(SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), val_df[val_df['skin_tone_class']==2], transform=val_transform), batch_size=32, shuffle=False),
                'tone 3': DataLoader(SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), val_df[val_df['skin_tone_class']==3], transform=val_transform), batch_size=32, shuffle=False),
                'tone 4': DataLoader(SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), val_df[val_df['skin_tone_class']==4], transform=val_transform), batch_size=32, shuffle=False),
                'tone 5': DataLoader(SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), val_df[val_df['skin_tone_class']==5], transform=val_transform), batch_size=32, shuffle=False)
            },
            'hair': {
                'level 1': DataLoader(SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), val_df[val_df['MONET_hair']<0.2], transform=val_transform), batch_size=32, shuffle=False),
                'level 2': DataLoader(SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), val_df[(val_df['MONET_hair']>=0.2) & (val_df['MONET_hair']<0.3)], transform=val_transform), batch_size=32, shuffle=False),
                'level 3': DataLoader(SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), val_df[(val_df['MONET_hair']>=0.3) & (val_df['MONET_hair']<0.5)], transform=val_transform), batch_size=32, shuffle=False),
                'level 4': DataLoader(SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), val_df[val_df['MONET_hair']>=0.5], transform=val_transform), batch_size=32, shuffle=False),
            },
            'drop': {
                'level 1': DataLoader(SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), val_df[val_df['MONET_gel_water_drop_fluid_dermoscopy_liquid']<0.3], transform=val_transform), batch_size=32, shuffle=False),
                'level 2': DataLoader(SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), val_df[(val_df['MONET_gel_water_drop_fluid_dermoscopy_liquid']>=0.3) & (val_df['MONET_gel_water_drop_fluid_dermoscopy_liquid']<0.4)], transform=val_transform), batch_size=32, shuffle=False),
                'level 3': DataLoader(SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), val_df[val_df['MONET_gel_water_drop_fluid_dermoscopy_liquid']>=0.4], transform=val_transform), batch_size=32, shuffle=False),
            },
            'ink': {
                'level 1': DataLoader(SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), val_df[
                    val_df['MONET_skin_markings_pen_ink_purple_pen'] < 0.2], transform=val_transform),
                                      batch_size=32, shuffle=False),
                'level 2': DataLoader(SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), val_df[
                    (val_df['MONET_skin_markings_pen_ink_purple_pen'] >= 0.2) & (
                                val_df['MONET_skin_markings_pen_ink_purple_pen'] < 0.4)],
                                                      transform=val_transform), batch_size=32, shuffle=False),

                'level 3': DataLoader(SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), val_df[
                    (val_df['MONET_skin_markings_pen_ink_purple_pen'] >= 0.4) & (
                            val_df['MONET_skin_markings_pen_ink_purple_pen'] < 0.6)],
                                                      transform=val_transform), batch_size=32, shuffle=False),
                'level 4': DataLoader(SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), val_df[
                    val_df['MONET_skin_markings_pen_ink_purple_pen'] >= 0.6], transform=val_transform),
                                      batch_size=32, shuffle=False),
            },
        }

        yield fold, train_dl, val_dl, test_dl


def load_isic_data_different_age_groups(data_path):
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),

        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),

        transforms.RandomApply([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            ),
            # v2.RandomChoice([v2.CutMix(num_classes=6), v2.MixUp(num_classes=6)])

        ], p=0.7),  # 70% chance to apply ONE OR MORE from the group

        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        ),
    ])

    train_df = pd.read_csv(os.path.join(data_path, 'train_6cls.csv'))
    val_df = pd.read_csv(os.path.join(data_path, 'val_6cls.csv'))

    train_dataset = SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), train_df[(train_df['age_approx']>=60) & (train_df['age_approx']<=75)], transform=train_transform)
    val_dataset = SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), val_df[(val_df['age_approx']>=60) & (val_df['age_approx']<=75)], transform=val_transform)

    train_dl = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=1, shuffle=False)

    age_35_dataset = SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), train_df[train_df['age_approx']==35], transform=val_transform)
    age_35_data_loader = DataLoader(age_35_dataset, batch_size=1, shuffle=False)
    age_30_dataset = SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), train_df[train_df['age_approx']<=30], transform=val_transform)
    age_30_data_loader = DataLoader(age_30_dataset, batch_size=1, shuffle=False)
    age_40_dataset = SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), train_df[train_df['age_approx']==40], transform=val_transform)
    age_40_data_loader = DataLoader(age_40_dataset, batch_size=1, shuffle=False)
    age_45_dataset = SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), train_df[train_df['age_approx']==45], transform=val_transform)
    age_45_data_loader = DataLoader(age_45_dataset, batch_size=1, shuffle=False)
    age_50_dataset = SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), train_df[train_df['age_approx']==50], transform=val_transform)
    age_50_data_loader = DataLoader(age_50_dataset, batch_size=1, shuffle=False)
    age_55_dataset = SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), train_df[train_df['age_approx']==55], transform=val_transform)
    age_55_data_loader = DataLoader(age_55_dataset, batch_size=1, shuffle=False)
    age_80_dataset = SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), train_df[train_df['age_approx']==80], transform=val_transform)
    age_80_data_loader = DataLoader(age_80_dataset, batch_size=1, shuffle=False)
    age_85_dataset = SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), train_df[train_df['age_approx']==85], transform=val_transform)
    age_85_data_loader = DataLoader(age_85_dataset, batch_size=1, shuffle=False)


    test_dl = {
        'age': {
            'under 30': age_30_data_loader,
            '35': age_35_data_loader,
            '40': age_40_data_loader,
            '45': age_45_data_loader,
            '50': age_50_data_loader,
            '55': age_55_data_loader,
            '80': age_80_data_loader,
            '85': age_85_data_loader,
        }
    }

    return train_dl, val_dl, test_dl

def load_isic_data_different_skin_tone_groups(data_path):
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),

        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),

        transforms.RandomApply([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            ),
            # v2.RandomChoice([v2.CutMix(num_classes=6), v2.MixUp(num_classes=6)])

        ], p=0.7),  # 70% chance to apply ONE OR MORE from the group

        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        ),
    ])

    train_df = pd.read_csv(os.path.join(data_path, 'train_6cls.csv'))
    val_df = pd.read_csv(os.path.join(data_path, 'val_6cls.csv'))

    train_dataset = SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), train_df[train_df['skin_tone_class']==3], transform=train_transform)
    val_dataset = SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), val_df[train_df['skin_tone_class']==3], transform=val_transform)

    train_dl = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=1, shuffle=False)

    skin_tone_1_dataset = SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), train_df[train_df['skin_tone_class']==1], transform=val_transform)
    skin_tone_1_loader = DataLoader(skin_tone_1_dataset, batch_size=1, shuffle=False)
    skin_tone_2_dataset = SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), train_df[train_df['skin_tone_class']==2], transform=val_transform)
    skin_tone_2_loader = DataLoader(skin_tone_2_dataset, batch_size=1, shuffle=False)
    skin_tone_4_dataset = SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), train_df[train_df['skin_tone_class']==4], transform=val_transform)
    skin_tone_4_loader = DataLoader(skin_tone_4_dataset, batch_size=1, shuffle=False)
    skin_tone_5_dataset = SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), train_df[train_df['skin_tone_class']==5], transform=val_transform)
    skin_tone_5_loader = DataLoader(skin_tone_5_dataset, batch_size=1, shuffle=False)


    test_dl = {
        'skin tone': {
            'tone 1': skin_tone_1_loader,
            'tone 2': skin_tone_2_loader,
            'tone 4': skin_tone_4_loader,
            'tone 5': skin_tone_5_loader,
        }
    }

    return train_dl, val_dl, test_dl


def load_isic_data_different_hair_groups(data_path):
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),

        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),

        transforms.RandomApply([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            ),
            # v2.RandomChoice([v2.CutMix(num_classes=6), v2.MixUp(num_classes=6)])

        ], p=0.7),  # 70% chance to apply ONE OR MORE from the group

        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        ),
    ])

    train_df = pd.read_csv(os.path.join(data_path, 'train_6cls.csv'))
    val_df = pd.read_csv(os.path.join(data_path, 'val_6cls.csv'))

    train_dataset = SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), train_df[train_df['MONET_hair']<0.2], transform=train_transform)
    val_dataset = SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), val_df[train_df['MONET_hair']<0.2], transform=val_transform)

    train_dl = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=1, shuffle=False)

    hairiness_1_dataset = SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), train_df[(train_df['MONET_hair']>=0.2) & (train_df['MONET_hair']<0.3)], transform=val_transform)
    hairiness_1_loader = DataLoader(hairiness_1_dataset, batch_size=1, shuffle=False)
    hairiness_2_dataset = SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), train_df[(train_df['MONET_hair']>=0.3) & (train_df['MONET_hair']<0.5)], transform=val_transform)
    hairiness_2_loader = DataLoader(hairiness_2_dataset, batch_size=1, shuffle=False)
    hairiness_3_dataset = SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), train_df[train_df['MONET_hair']>=0.5], transform=val_transform)
    hairiness_3_loader = DataLoader(hairiness_3_dataset, batch_size=1, shuffle=False)


    test_dl = {
        'hairiness': {
            'level 1': hairiness_1_loader,
            'level 2': hairiness_2_loader,
            'level 3': hairiness_3_loader
        }
    }

    return train_dl, val_dl, test_dl

def load_data_different_training_size(data_path):
    train_loaders = {}
    for i in range(10,110, 10):
        train_dataset = MorphoMNISTDataset(os.path.join(data_path,'plain', 'train-images-idx3-ubyte.gz'), os.path.join(data_path,'plain', 'train-labels-idx1-ubyte.gz'), portion=i)
        train_dl = DataLoader(train_dataset, batch_size=64, shuffle=True)
        train_loaders[i] = train_dl

    val_dataset = MorphoMNISTDataset(os.path.join(data_path,'plain', 't10k-images-idx3-ubyte.gz'), os.path.join(data_path,'plain', 't10k-labels-idx1-ubyte.gz'))
    thining5_dataset = MorphoMNISTDataset(os.path.join(data_path,'thinning', 't10k-images-idx3-ubyte-7.gz'), os.path.join(data_path,'plain', 't10k-labels-idx1-ubyte.gz'))

    val_dl = DataLoader(val_dataset, batch_size=1, shuffle=False)
    thining5_dl = DataLoader(thining5_dataset, batch_size=1, shuffle=False)

    return train_loaders, val_dl, thining5_dl

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

def load_data_blurred(data_path):
    transform = transforms.ToTensor()
    plain_images = pars_gzip_file(os.path.join(data_path, 'plain', 'train-images-idx3-ubyte.gz'))
    blur_low_sever_images = pars_gzip_file(os.path.join(data_path, 'blured', 'train-images-blur-ks(5, 5)-s[1.0, 1.0].gz'))
    blur_mid_sever_images = pars_gzip_file(os.path.join(data_path, 'blured', 'train-images-blur-ks(11, 11)-s[2.0, 2.0].gz'))
    blur_high_sever_images = pars_gzip_file(os.path.join(data_path, 'blured', 'train-images-blur-ks(17, 17)-s[3.0, 3.0].gz'))
    labels = pars_gzip_file(os.path.join(data_path, 'plain', 'train-labels-idx1-ubyte.gz'))

    n = len(labels)
    n_portion = n//4
    train_images= np.concatenate([plain_images[0:n_portion], blur_low_sever_images[n_portion:2*n_portion],blur_mid_sever_images[2*n_portion:3*n_portion],blur_high_sever_images[3*n_portion:4*n_portion]])
    train_dataset = MorphoMNISTDataset(train_images, labels[0:4*n_portion])
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    val_loaders = {}
    plain_val_dataset = MorphoMNISTDataset(os.path.join(data_path, 'plain', 't10k-images-idx3-ubyte.gz'),
                                     os.path.join(data_path, 'plain', 't10k-labels-idx1-ubyte.gz'))
    val_loaders['plain'] = DataLoader(plain_val_dataset, batch_size=64, shuffle=True)
    blured_val_dataset = MorphoMNISTDataset(os.path.join(data_path, 'blured', 'val-images-blur-ks(17, 17)-s[3.0, 3.0].gz'),
                                            os.path.join(data_path, 'blured', 'val-labels-blur-ks(17, 17)-s[3.0, 3.0].gz'))
    val_loaders['high_sever'] = DataLoader(blured_val_dataset, batch_size=64, shuffle=False)

    blured_val_dataset = MorphoMNISTDataset(
        os.path.join(data_path, 'blured', 'val-images-blur-ks(11, 11)-s[2.0, 2.0].gz'),
        os.path.join(data_path, 'blured', 'val-labels-blur-ks(11, 11)-s[2.0, 2.0].gz'))
    val_loaders['med_sever'] = DataLoader(blured_val_dataset, batch_size=64, shuffle=False)

    blured_val_dataset = MorphoMNISTDataset(
        os.path.join(data_path, 'blured', 'val-images-blur-ks(5, 5)-s[1.0, 1.0].gz'),
        os.path.join(data_path, 'blured', 'val-labels-blur-ks(5, 5)-s[1.0, 1.0].gz'))
    val_loaders['low_sever'] = DataLoader(blured_val_dataset, batch_size=64, shuffle=False)

    img1 = pars_gzip_file(os.path.join(data_path, 'plain', 't10k-images-idx3-ubyte.gz'))
    img2 = pars_gzip_file(os.path.join(data_path, 'blured', 'val-images-blur-ks(5, 5)-s[1.0, 1.0].gz'))
    img3 = pars_gzip_file(os.path.join(data_path, 'blured', 'val-images-blur-ks(11, 11)-s[2.0, 2.0].gz'))
    img4 = pars_gzip_file(os.path.join(data_path, 'blured', 'val-images-blur-ks(17, 17)-s[3.0, 3.0].gz'))
    lbl = pars_gzip_file(os.path.join(data_path, 'plain', 't10k-labels-idx1-ubyte.gz'))
    n = len(lbl)
    n_portion = n//4
    img = np.concatenate((img1[0:n_portion], img2[n_portion:2*n_portion], img3[2*n_portion:3*n_portion], img4[3*n_portion:4*n_portion]))
    common_valid_dataset = MorphoMNISTDataset(img, lbl[0:4*n_portion])
    common_valid_loader = DataLoader(common_valid_dataset, batch_size=64, shuffle=False)

    return train_loader, common_valid_loader, val_loaders

def load_aleatoric_data3(data_path):
    train_loaders = {}
    val_loaders = {}
    for p in np.arange(0.0, 1.1, 0.3):
        name = f"blur_{int(p*100)}%"
        train_dataset = MorphoMNISTDataset(
            os.path.join(data_path, 'blured', f'train-images-{name }.gz'),
            os.path.join(data_path, 'blured', f'train-labels-{name }.gz') )
        val_dataset = MorphoMNISTDataset(
            os.path.join(data_path, 'blured', f'val-images-{name }.gz'),
            os.path.join(data_path, 'blured', f'val-labels-{name }.gz'))

        name = name.split('_')[-1].split('.')[0]
        train_loaders[name] = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loaders[name] = DataLoader(val_dataset, batch_size=1, shuffle=False)

    return train_loaders, val_loaders


def load_isic_data_skin_tone(data_path):

    densenet_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train_gt_file = pd.read_csv(os.path.join(data_path, 'ISIC-2017_Training_Part3_GroundTruth.csv'))
    train_dataset = SkinISICDataset(os.path.join(data_path, 'ISIC-2017_Training_Data'), train_gt_file, transform=densenet_transform)
    valid_gt_file = pd.read_csv(os.path.join(data_path, 'ISIC-2017_Validation_Part3_GroundTruth.csv'))
    valid_dataset = SkinISICDataset(os.path.join(data_path, 'ISIC-2017_Validation_Data'), valid_gt_file, transform=densenet_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    test_gt_file = pd.read_csv(os.path.join(data_path, 'ISIC-2017_Test_v2_Part3_GroundTruth.csv'))
    skin_tone_file = pd.read_csv(os.path.join(data_path, 'skin_tone_test_no_hair.csv'))
    skin_tone_file['image_id'] = skin_tone_file['image'].map(lambda x: os.path.basename(x)[:-4])
    test_data = pd.merge(test_gt_file, skin_tone_file,  on='image_id')
    skin_tones = pd.unique(test_data['skin_tone'])
    test_loader = {}
    for tone in skin_tones:
        tone_data = test_data[test_data['skin_tone'] == tone]
        test_dataset = SkinISICDataset(os.path.join(data_path, 'ISIC-2017_Test_v2_Data'), tone_data, transform=densenet_transform)
        test_loader[tone] = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, valid_loader, test_loader


def load_isic_data_hair_level(data_path):

    densenet_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train_gt_file = pd.read_csv(os.path.join(data_path, 'ISIC-2017_Training_Part3_GroundTruth.csv'))
    train_dataset = SkinISICDataset(os.path.join(data_path, 'ISIC-2017_Training_Data'), train_gt_file, transform=densenet_transform)
    valid_gt_file = pd.read_csv(os.path.join(data_path, 'ISIC-2017_Validation_Part3_GroundTruth.csv'))
    valid_dataset = SkinISICDataset(os.path.join(data_path, 'ISIC-2017_Validation_Data'), valid_gt_file, transform=densenet_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    test_gt_file = pd.read_csv(os.path.join(data_path, 'ISIC-2017_Test_v2_Part3_GroundTruth.csv'))
    hairiness_file = pd.read_csv(os.path.join(data_path, 'hairiness_test.csv'))
    hairiness_file['image_id'] = hairiness_file['image'].map(lambda x: os.path.basename(x)[:-4])
    test_data = pd.merge(test_gt_file, hairiness_file,  on='image_id')
    hairiness_levels = pd.unique(test_data['hairiness_level'])
    test_loader = {}
    for tone in hairiness_levels:
        tone_data = test_data[test_data['hairiness_level'] == tone]
        test_dataset = SkinISICDataset(os.path.join(data_path, 'ISIC-2017_Test_v2_Data'), tone_data, transform=densenet_transform)
        test_loader[f'hairiness_level_{tone}'] = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, valid_loader, test_loader


def load_isis_data_difficulty_level(data_path):
    train_loaders = {}
    val_loaders = {}

    densenet_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    difficulty_levels = ['single image expert consensus', 'serial imaging showing no change', 'histopathology']
    test_gt_file = pd.read_csv(os.path.join(data_path, 'ISIC2018_Task3_Training_GroundTruth.csv'))
    meta_data  = pd.read_csv(os.path.join(data_path, 'ISIC2018_Task3_Training_LesionGroupings.csv'))
    test_data = pd.merge(test_gt_file, meta_data, on='image')
    test_data = test_data.rename(columns={'image':'image_id'})
    classes = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    test_data['label'] = test_data[classes].values.argmax(axis=1)
    for diff in difficulty_levels:
        diff_data = test_data[test_data['diagnosis_confirm_type'] == diff]
        train, test = train_test_split(diff_data, test_size=0.1)
        train_dataset = SkinISICDataset(os.path.join(data_path, 'ISIC2018_Task3_Training_Input'), train,
                                       transform=densenet_transform)
        val_dataset = SkinISICDataset(os.path.join(data_path, 'ISIC2018_Task3_Training_Input'), test, transform=densenet_transform)
        train_loaders[diff] = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loaders[diff] = DataLoader(val_dataset, batch_size=1, shuffle=False)

    return train_loaders, val_loaders

def load_isic_skin_tone_training_size(data_path):
    densenet_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train_loaders = {}
    metadata = pd.read_csv(os.path.join(data_path,'MILK10k_Training_Metadata.csv'))
    metadata = metadata.drop_duplicates('lesion_id')
    gt = pd.read_csv(os.path.join(data_path,'MILK10k_Training_GroundTruth.csv'))
    selected_classes = ['BCC', 'SCCKA', 'AKIEC', 'NV', 'BKL', 'MEL']
    classes = ['AKIEC', 'BCC', 'BEN_OTH', 'BKL', 'DF', 'INF', 'MAL_OTH', 'MEL', 'NV', 'SCCKA', 'VASC']
    gt['label_cat'] = gt[classes].idxmax(axis=1)
    gt = gt[gt['label_cat'].isin(selected_classes)]
    full_data = pd.merge(gt, metadata, on='lesion_id')
    full_data['image_id'] = full_data.apply(lambda x: x['lesion_id'] + '/' + x['isic_id'], axis=1)
    full_data['label'] = gt[selected_classes].values.argmax(axis=1)
    in_dist = full_data[full_data['skin_tone_class']==3]
    out_of_dist = full_data[full_data['skin_tone_class']==4]
    train, val = train_test_split(in_dist, test_size=0.1)

    num_divide = 3
    df = []
    for i in range(num_divide):
        df.append(pd.DataFrame())
    for cls in selected_classes:
        class_data = train[train['label_cat'] == cls].sample(frac=1).reset_index(drop=True)
        n = len(class_data)
        for i in range(num_divide):
            df[i] = pd.concat([df[i], class_data[i*n//num_divide:(i+1)*n//num_divide]], ignore_index=True)

    for i in range(3):
        train_dataset = SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), pd.concat(df[:i+1]),
                                       transform=densenet_transform)
        train_loaders[i+1] = DataLoader(train_dataset, batch_size=64, shuffle=True)

    val_dataset = SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), val, transform=densenet_transform)
    val_dl = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_dataset = SkinISICDataset(os.path.join(data_path, 'MILK10k_Training_Input'), out_of_dist, transform=densenet_transform)
    test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loaders, val_dl, test_dl
