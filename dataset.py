import os
import csv
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from utils import labels as get_labels


class FER2013Dataset(Dataset):
    """Loads FER2013 from CSV or folder structure.

    Supports either:
    - data/fer2013/fer2013.csv (CSV with "pixels" column and "Usage" values)
    - data/fer2013/<train|val|test>/<label>/*.jpg
    - data/fer2013/processed/<split>/*.png
    """

    def __init__(self, root='data/fer2013', split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        csv_path = os.path.join(root, 'fer2013.csv')
        processed_dir = os.path.join(root, 'processed', split)
        split_dir = os.path.join(root, split)

        self.samples = []

        if os.path.isfile(csv_path):
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    usage = row.get('Usage', '').lower()
                    # map common usage names to splits
                    if (split == 'train' and 'training' in usage) or (split == 'val' and 'public' in usage) or (split == 'test' and 'private' in usage):
                        pixels = np.fromstring(row['pixels'], sep=' ', dtype=np.uint8)
                        self.samples.append((int(row['emotion']), pixels.reshape(48, 48)))
        elif os.path.isdir(processed_dir):
            for fname in os.listdir(processed_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    label = fname.split('_')[0]
                    path = os.path.join(processed_dir, fname)
                    self.samples.append((label, path))
        elif os.path.isdir(split_dir):
          
            for label in sorted(os.listdir(split_dir)):
                label_dir = os.path.join(split_dir, label)
                if not os.path.isdir(label_dir):
                    continue
                for fname in os.listdir(label_dir):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        path = os.path.join(label_dir, fname)
                        self.samples.append((label, path))
        else:
            raise FileNotFoundError('Could not find fer2013.csv or processed images in %s' % root)

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(),
                transforms.Resize((48, 48)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])

        
        canonical = {k.lower(): v for v, k in enumerate(get_labels())}
        self.label_map = canonical

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label, data = self.samples[idx]
        if isinstance(data, str):
            img = Image.open(data).convert('L')
            img = np.array(img, dtype=np.uint8)
        else:
            img = data.astype(np.uint8)

        # ensure single channel HxW
        if img.ndim == 3:
            img = img[:, :, 0]

        img_t = self.transform(img)

        # convert folder-name labels to index if needed
        if isinstance(label, str):
            key = label.lower()
            if key in self.label_map:
                lbl = int(self.label_map[key])
            else:
                # fallback: try to parse numeric
                try:
                    lbl = int(label)
                except Exception:
                    raise ValueError(f'Unknown label {label} and not in canonical labels')
        else:
            lbl = int(label)

        return lbl, img_t


if __name__ == '__main__':
    ds = FER2013Dataset(root='data/fer2013', split='train')
    print('Loaded', len(ds), 'samples')
