import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

threshold = 1
table = []
for i in range(256):
    if i < threshold:
        table.append(0)
    else:
        table.append(1)


class HelaDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(HelaDataset, self).__init__()
        self.flag = "training" if train else "test"
        data_root = os.path.join(root, "DIC-C2DH-HeLa1", self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".tif")]
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]

        self.mask = [os.path.join(data_root, "mask", "man_seg" + i.split("t")[1]+"tif")
                     for i in img_names]
        # check files
        for i in self.mask:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        mask = Image.open(self.mask[idx]).convert('L')
        mask = mask.point(table, "1")
        # mask = np.array(mask) / 255
        # # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        # mask = Image.fromarray(mask)
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)
        return img, mask

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


# root_path = r"D:\dataset"
# datasets1 = HelaDataset(root_path,train=True)

