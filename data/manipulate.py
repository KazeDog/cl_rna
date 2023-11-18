import torch
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset

from .dataloader import RNAdata


class SubDataset(Dataset):
    '''To sub-sample a dataset, taking only those samples with label in [sub_labels].

    After this selection of samples has been made, it is possible to transform the target-labels,
    which can be useful when doing continual learning with fixed number of output units.'''

    def __init__(self, datadir, sub_labels, seqlen):
        super().__init__()
        # self.datadir = datadir
        self.datalist = []
        self.sub_indeces = []
        print('sub_labels=', sub_labels)
        for label in sub_labels:
            self.datalist.append(RNAdata(datadir + '/' + f'{label}.csv', seqlen))
        self.dataset = ConcatDataset(self.datalist)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        return sample


## memory list
class MemorySetDataset(Dataset):
    '''Create dataset from list of <np.arrays> with shape (N, C, H, W) (i.e., with N images each).

    The images at the i-th entry of [memory_sets] belong to class [i], unless a [target_transform] is specified'''

    def __init__(self, memory_sets, target_transform=None):
        super().__init__()
        self.memory_sets = memory_sets
        self.target_transform = target_transform

    def __len__(self):
        total = 0
        for class_id in range(len(self.memory_sets)):
            total += len(self.memory_sets[class_id])
        return total

    def __getitem__(self, index):
        total = 0
        for class_id in range(len(self.memory_sets)):
            # print('class_id=', class_id)
            examples_in_this_class = len(self.memory_sets[class_id])
            if index < (total + examples_in_this_class):
                # class_id_to_return = class_id if self.target_transform is None else self.target_transform(class_id)
                class_id_to_return = class_id // 2 + 1 if class_id % 2 != 0 else 0
                example_id = index - total
                break
            else:
                total += examples_in_this_class
        image = torch.from_numpy(self.memory_sets[class_id][example_id])
        class_id_to_return = torch.tensor(class_id_to_return)
        return (image, class_id_to_return)