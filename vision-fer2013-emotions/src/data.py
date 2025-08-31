from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def _t_train(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])

def _t_eval(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

def make_train_val_loaders(root="data/fer", img_size=48, batch_size=64):
    train_ds = datasets.ImageFolder(f"{root}/train", transform=_t_train(img_size))
    val_ds   = datasets.ImageFolder(f"{root}/val",   transform=_t_eval(img_size))
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0),
            DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0),
            train_ds.class_to_idx)

def make_test_loader(root="data/fer", img_size=48, batch_size=64):
    test_ds = datasets.ImageFolder(f"{root}/test", transform=_t_eval(img_size))
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0), test_ds.class_to_idx
