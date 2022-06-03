import os
import warnings
import pandas as pd

import PIL
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

PIL.Image.MAX_IMAGE_PIXELS = 933120000


def get_dataset(dataset):

    if dataset == "wikiart"                 : return WikiArtDataset
    if dataset == "multitask_painting_100k" : return MultitaskPainting100kDataset


class WikiArtDataset(Dataset):

    num_styles = 27
    num_genres = 10

    def __init__(self, root_dir="/data/wikiart", split="train", transform=None):
        warnings.filterwarnings(action="ignore", category=Image.DecompressionBombWarning)
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        self.images = []
        self.styles = []
        self.genres = []

        img_dir = os.path.join(root_dir, "images")
        csv_file = os.path.join(root_dir, "labels", f"{split}.csv")

        df = pd.read_csv(csv_file)
        self.images = (img_dir + "/" + df.path).tolist()
        self.styles = df.style_id.tolist()
        self.genres = df.genre_id.tolist()

        if transform is None:
            if split == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize((252, 252)),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform

    def __getitem__(self, item):
        img = Image.open(self.images[item])
        img = self.transform(img)
        return img, self.styles[item], self.genres[item]

    def __len__(self):
        return len(self.styles)


class MultitaskPainting100kDataset(Dataset):

    num_styles = 125
    num_genres = 41

    def __init__(self, root_dir="/data/multitask_painting_100k", split="train", transform=None):
        warnings.filterwarnings(action="ignore", category=Image.DecompressionBombWarning)
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        self.images = []
        self.styles = []
        self.genres = []

        img_dir = os.path.join(root_dir, "images")
        csv_file = os.path.join(root_dir, 'labels', f"{split}.csv")

        df = pd.read_csv(csv_file)
        self.images = (img_dir + "/" + df.path).tolist()
        self.styles = df.style_id.tolist()
        self.genres = df.genre_id.tolist()

        if transform is None:
            if split == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize((252, 252)),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform

    def __getitem__(self, item):
        img = Image.open(self.images[item]).convert("RGB")
        img = self.transform(img)
        return img, self.styles[item], self.genres[item]

    def __len__(self):
        return len(self.styles)
