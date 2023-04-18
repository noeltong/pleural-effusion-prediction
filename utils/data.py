import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
import os
from sklearn.model_selection import train_test_split
from torch.utils.data.distributed import DistributedSampler


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_target = self.next_target.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target


class XRayDataset(Dataset):
    def __init__(self, images, targets, mode='train') -> None:
        super().__init__()
        self.images = images
        self.targets = targets
        self.aug_fn = transforms.Compose([
            transforms.RandomCrop(45),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.GaussianBlur(5, sigma=(0.1, 0.5)),
        ])

        self.mode = mode

    def __getitem__(self, index):
        img = self.images[index].squeeze().unsqueeze(0)
        # in shape of [B, 1, H, W]
        target = self.targets[index].squeeze()

        if self.mode == 'train':
            img = self.aug_fn(img)

        return img, target

    def __len__(self):
        return self.images.shape[0]


def get_dataloader(config):

    data = np.load(os.path.join(config.data.path, 'images.npy'))
    targets = np.load(os.path.join(config.data.path, 'targets.npy'))

    X_train, X_test, y_train, y_test = train_test_split(
        data, targets, test_size=0.25, random_state=config.seed
    )

    X_train = torch.asarray(X_train).float()
    X_test = torch.asarray(X_test).float()
    y_train = torch.asarray(y_train).float()
    y_test = torch.asarray(y_test).float()

    train_dataset = XRayDataset(X_train, y_train, mode='train')
    test_dataset = XRayDataset(X_test, y_test, mode='test')

    train_sampler = DistributedSampler(train_dataset)
    test_sampler = DistributedSampler(test_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        sampler=train_sampler,
        drop_last=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        prefetch_factor=config.data.prefetch_factor
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        sampler=test_sampler,
        drop_last=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        prefetch_factor=config.data.prefetch_factor
    )

    return train_loader, test_loader, train_sampler, test_sampler
