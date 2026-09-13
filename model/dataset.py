import string

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

EMNIST_MEAN = 0.1307
EMNIST_STD = 0.3081

# EMNIST Balanced label order, per the official mapping file.
BALANCED_CLASSES = (
    list(string.digits)
    + list(string.ascii_uppercase)
    + ["a", "b", "d", "e", "f", "g", "h", "n", "q", "r", "t"]
)
assert len(BALANCED_CLASSES) == 47


def label_to_char(label: int) -> str:
    return BALANCED_CLASSES[label]


train_transform = transforms.Compose(
    [
        transforms.RandomRotation(8),          # small jitter, real handwriting varies
        transforms.RandomAffine(0, translate=(0.08, 0.08), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((EMNIST_MEAN,), (EMNIST_STD,)),
    ]
)

eval_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((EMNIST_MEAN,), (EMNIST_STD,)),
    ]
)


def get_loaders(data_dir: str, batch_size: int = 128, num_workers: int = 2):
    train_set = datasets.EMNIST(
        data_dir, split="balanced", train=True, download=True, transform=train_transform
    )
    test_set = datasets.EMNIST(
        data_dir, split="balanced", train=False, download=True, transform=eval_transform
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader
