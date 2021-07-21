from typing import Callable, Tuple

import pathlib

import h5py
import torch
from torch.utils.data import Dataset


class OnePersonDataset(Dataset):
    def __init__(self, person_id_str: str, dataset_path: pathlib.Path,
                 transform: Callable):
        self.transform = transform

        # In case of the MPIIGaze dataset, each image is so small that
        # reading image will become a bottleneck even with HDF5.
        # So, first load them all into memory.
        with h5py.File(dataset_path, 'r') as f:
            images = f.get(f'{person_id_str}/image')[()]
            poses = f.get(f'{person_id_str}/pose')[()]
            #gazes = f.get(f'{person_id_str}/gaze')[()]
            targets = f.get(f'{person_id_str}/target')[()]
        assert len(images) == 3000
        assert len(poses) == 3000
        #assert len(gazes) == 3000
        assert len(targets) == 3000
        self.images = images
        self.poses = poses
        #self.gazes = gazes
        self.targets = targets

    def __getitem__(self, index: int
                    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        image = self.transform(self.images[index])
        pose = torch.from_numpy(self.poses[index])
        #gaze = torch.from_numpy(self.gazes[index])
        target = torch.from_numpy(self.targets[index])
        return image, pose, target
        #return image,gaze,target

    def __len__(self) -> int:
        return len(self.images)
