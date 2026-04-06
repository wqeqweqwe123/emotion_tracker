import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import FER2013Dataset
from model import SimpleFERNet
from utils import ensure_models_dir


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for labels, imgs in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for labels, imgs in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/fer2013', help='data folder')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save', default='models/best_model.pth')
    parser.add_argument('--resume', default='', help='path to checkpoint to resume from')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # use folder-based splits if available: 'train' and 'val' (fallback to 'test')
    train_split = 'train'
    val_split = 'val' if os.path.isdir(os.path.join(args.data, 'val')) else 'test'
    train_ds = FER2013Dataset(root=args.data, split=train_split)
    val_ds = FER2013Dataset(root=args.data, split=val_split)

    # use num_workers=0 for better compatibility on macOS and to avoid multiprocessing shutdown issues
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = SimpleFERNet()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 1
    # option to resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('Loading checkpoint', args.resume)
            ckpt = torch.load(args.resume, map_location=device)
            # load model weights
            if 'state_dict' in ckpt:
                model.load_state_dict(ckpt['state_dict'])
            else:
                model.load_state_dict(ckpt)
            # load optimizer state if available
            if 'optimizer' in ckpt and ckpt['optimizer'] is not None:
                try:
                    optimizer.load_state_dict(ckpt['optimizer'])
                except Exception:
                    print('Warning: could not load optimizer state (optimizer mismatch)')
            # resume epoch
            if 'epoch' in ckpt:
                start_epoch = int(ckpt['epoch']) + 1
            # best_acc
            best_acc = float(ckpt.get('best_acc', 0.0))
            print(f'Resuming from epoch {start_epoch}, best_acc={best_acc}')
        else:
            print('Resume checkpoint not found:', args.resume)

    ensure_models_dir(os.path.dirname(args.save) or 'models')

    best_acc = 0.0
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = val_epoch(model, val_loader, criterion, device)
            print(f'Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}')

            # save per-epoch checkpoint (allow resume)
            epoch_ckpt_path = os.path.join(os.path.dirname(args.save) or 'models', f'epoch_{epoch}.pth')
            try:
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'best_acc': best_acc}, epoch_ckpt_path)
            except Exception as e:
                print('Warning: failed to save epoch checkpoint:', e)

            if val_acc > best_acc:
                best_acc = val_acc
                try:
                    torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'best_acc': best_acc}, args.save)
                except Exception:
                    # fallback to saving only state_dict
                    torch.save({'state_dict': model.state_dict(), 'best_acc': best_acc}, args.save)
                print('Saved best model', args.save)
    except KeyboardInterrupt:
        print('\nTraining interrupted by user. Saving current model...')
        try:
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'best_acc': best_acc}, args.save)
            print('Saved model to', args.save)
        except Exception as e:
            print('Failed to save model:', e)
        raise


if __name__ == '__main__':
    main()
