import torch
import torchvision
from torchvision import transforms

# 准备训练数据和测试数据
def prepare_dataloader(num_workers=8, train_batch_size=128, eval_batch_size=256):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_set = torchvision.datasets.CIFAR10(
        root="data", train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root="data", train=False, download=True, transform=test_transform
    )
    train_sampler = torch.utils.data.RandomSampler(train_set)
    test_sampler = torch.utils.data.SequentialSampler(test_set)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=train_batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=eval_batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
    )
    return train_loader, test_loader