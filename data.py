from config import *
from easydl import FileListDataset
from collections import Counter
from torchvision.transforms.transforms import *
from torch.utils.data import DataLoader, WeightedRandomSampler

'''
assume classes across domains are the same.
[0 1 ..................................................................... N - 1]
|----common classes --||----source private classes --||----target private classes --|
'''
a, b, c = args.data.dataset.n_share, args.data.dataset.n_source_private, args.data.dataset.n_total
c = c - a - b
common_classes = [i for i in range(a)]
source_private_classes = [i + a for i in range(b)]
target_private_classes = [i + a + b for i in range(c)]

source_classes = common_classes + source_private_classes
target_classes = common_classes + target_private_classes

# target-private label
tp_classes = sorted(set(target_classes) - set(source_classes))
# source-private label
sp_classes = sorted(set(source_classes) - set(target_classes))
# common label
common_classes = sorted(set(source_classes) - set(sp_classes))

classes_set = {
    'source_classes': source_classes,
    'target_classes': target_classes,
    'tp_classes': tp_classes,
    'sp_classes': sp_classes,
    'common_classes': common_classes
}

uniformed_index = len(classes_set['source_classes'])

train_transform = Compose([
    Scale((256, 256)),
    RandomCrop(224),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = Compose([
    Scale((256, 256)),
    CenterCrop(224),
    ToTensor(),
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

normalize = Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])

# FileListDataset
source_train_ds = FileListDataset(list_path=source_txt, path_prefix=dataset_root,
                            transform=train_transform, return_id=True, filter=(lambda x: x in source_classes))
target_train_ds = FileListDataset(list_path=target_txt, path_prefix=dataset_root,
                            transform=train_transform, return_id=True, filter=(lambda x: x in target_classes))
source_test_ds = FileListDataset(list_path=source_txt, path_prefix=dataset_root,
                            transform=test_transform, return_id=False, filter=(lambda x: x in source_classes))
target_test_ds = FileListDataset(list_path=target_txt, path_prefix=dataset_root,
                            transform=test_transform, return_id=False, filter=(lambda x: x in target_classes))

# balanced sampler for source train
classes = source_train_ds.labels
freq = Counter(classes)
class_weight = {x : 1.0 / freq[x] if args.data.dataloader.class_balance else 1.0 for x in freq}

source_weights = [class_weight[x] for x in source_train_ds.labels]
sampler = WeightedRandomSampler(source_weights, len(source_train_ds.labels))

source_train_dl = DataLoader(dataset=source_train_ds, batch_size=batch_size,
                             sampler=sampler, num_workers=num_workers, drop_last=True)
target_train_dl = DataLoader(dataset=target_train_ds, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, drop_last=True)
target_test_dl = DataLoader(dataset=target_test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, drop_last=False)
# for memory queue init
target_initMQ_dl = DataLoader(dataset=target_train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, drop_last=True)
# for tsne feature visualization
source_test_dl = DataLoader(dataset=source_test_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=False)