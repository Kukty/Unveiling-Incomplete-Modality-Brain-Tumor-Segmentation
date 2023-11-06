# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
import os

import numpy as np
import torch

from monai import data, transforms
from typing import Tuple

def mask_rand_patch(
    window_sizes: Tuple[int, int, int], input_sizes: Tuple[int, int, int], mask_ratio: float, samples: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Patch-wise random masking."""
    if len(window_sizes) != len(input_sizes) or any(
        [input_size % window_size != 0 for window_size, input_size in zip(window_sizes, input_sizes)]
    ):
        raise ValueError(f"{window_sizes} & {input_sizes} is not compatible.")

    mask_shape = [input_size // window_size for input_size, window_size in zip(input_sizes, window_sizes)]
    num_patches = np.prod(mask_shape).item()
    mask = np.ones(num_patches, dtype=bool)
    indices = np.random.choice(num_patches, round(num_patches * mask_ratio), replace=False)
    mask[indices] = False
    mask = mask.reshape(mask_shape)
    wh, ww, wd = window_sizes
    mask = np.logical_or(mask[:, None, :, None, :, None], np.zeros([1, wh, 1, ww, 1, wd], dtype=bool)).reshape(
        input_sizes
    )
    mask = torch.from_numpy(mask).to(samples.device)

    res = samples.detach().clone()
    res[:, :, mask] = 0
    return res, mask


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def datafold_read(datalist, basedir, fold=0, key="training",modalities = ["Flair", "T1", "T1c" ,"T2"]):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    # delay on modalities to upadate json_data 
    # for item in json_data:
    #     if "Flair" not in modalities:
    #         item['image'] = [image for image in item['image'] if not image.endswith('_flair.nii.gz')]
    #     if "T1" not in modalities:
    #         item['image'] = [image for image in item['image'] if not image.endswith('_t1.nii.gz')]
    #     if "T1c" not in modalities:
    #         item['image'] = [image for image in item['image'] if not image.endswith('_t1ce.nii.gz')]
    #     if "T2" not in modalities:
    #         item['image'] = [image for image in item['image'] if not image.endswith('_t2.nii.gz')]
    
    for d in json_data:
        for k, v in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val

class EnsureSingleChannel:
    def __init__(self, key):
        self.key = key

    def __call__(self, data):
        image = data[self.key]
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)
        data[self.key] = image
        return data
    
def get_loader(args):
    data_dir = args.data_dir
    datalist_json = args.json_list
    train_files, validation_files = datafold_read(datalist=datalist_json, basedir=data_dir, fold=args.fold,modalities = args.in_modalities)
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            # EnsureSingleChannel(key="image"),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.CropForegroundd(
                keys=["image", "label"], source_key="image", k_divisible=[args.roi_x, args.roi_y, args.roi_z]
            ),
            transforms.RandSpatialCropd(
                keys=["image", "label"], roi_size=[args.roi_x, args.roi_y, args.roi_z], random_size=False
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    if args.test_mode:
        val_ds = data.Dataset(data=validation_files, transform=test_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
        )

        loader = test_loader
    else:
        train_ds = data.Dataset(data=train_files, transform=train_transform)

        train_sampler = Sampler(train_ds) if args.distributed else None
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
        )
        val_ds = data.Dataset(data=validation_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
        )
        loader = [train_loader, val_loader]

    return loader
