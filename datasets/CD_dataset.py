# """
# 变化检测数据集
# """
#
# import os
# from PIL import Image
# import numpy as np
#
# from torch.utils import data
#
# from datasets.data_utils import CDDataAugmentation
#
#
# """
# CD data set with pixel-level labels；
# ├─image
# ├─image_post
# ├─label
# └─list
# """
# IMG_FOLDER_NAME = "A"
# IMG_POST_FOLDER_NAME = 'B'
# LIST_FOLDER_NAME = 'list'
# ANNOT_FOLDER_NAME = "label"
#
# IGNORE = 255
#
# label_suffix='.png' # jpg for gan dataset, others : png
#
# def load_img_name_list(dataset_path):
#     img_name_list = np.loadtxt(dataset_path, dtype=np.str_)
#     if img_name_list.ndim == 2:
#         return img_name_list[:, 0]
#     return img_name_list
#
#
# def load_image_label_list_from_npy(npy_path, img_name_list):
#     cls_labels_dict = np.load(npy_path, allow_pickle=True).item()
#     return [cls_labels_dict[img_name] for img_name in img_name_list]
#
#
# def get_img_post_path(root_dir,img_name):
#     return os.path.join(root_dir, IMG_POST_FOLDER_NAME, img_name)
#
#
# def get_img_path(root_dir, img_name):
#     return os.path.join(root_dir, IMG_FOLDER_NAME, img_name)
#
#
# def get_label_path(root_dir, img_name):
#     return os.path.join(root_dir, ANNOT_FOLDER_NAME, img_name.replace('.jpg', label_suffix))
#
#
# class ImageDataset(data.Dataset):
#     """VOCdataloder"""
#     def __init__(self, root_dir, split='train', img_size=256, is_train=True,to_tensor=True):
#         super(ImageDataset, self).__init__()
#         self.root_dir = root_dir
#         self.img_size = img_size
#         self.split = split  # train | train_aug | val
#         # self.list_path = self.root_dir + '/' + LIST_FOLDER_NAME + '/' + self.list + '.txt'
#         self.list_path = os.path.join(self.root_dir, LIST_FOLDER_NAME, self.split+'.txt')
#         self.img_name_list = load_img_name_list(self.list_path)
#
#         self.A_size = len(self.img_name_list)  # get the size of dataset A
#         self.to_tensor = to_tensor
#         if is_train:
#             self.augm = CDDataAugmentation(
#                 img_size=self.img_size,
#                 with_random_hflip=True,
#                 with_random_vflip=True,
#                 with_scale_random_crop=True,
#                 with_random_blur=True,
#             )
#         else:
#             self.augm = CDDataAugmentation(
#                 img_size=self.img_size
#             )
#     def __getitem__(self, index):
#         name = self.img_name_list[index]
#         A_path = get_img_path(self.root_dir, self.img_name_list[index % self.A_size])
#         B_path = get_img_post_path(self.root_dir, self.img_name_list[index % self.A_size])
#
#         img = np.asarray(Image.open(A_path).convert('RGB'))
#         img_B = np.asarray(Image.open(B_path).convert('RGB'))
#
#         [img, img_B], _ = self.augm.transform([img, img_B],[], to_tensor=self.to_tensor)
#
#         return {'A': img, 'B': img_B, 'name': name}
#
#     def __len__(self):
#         """Return the total number of images in the dataset."""
#         return self.A_size
#
#
# class CDDataset(ImageDataset):
#
#     def __init__(self, root_dir, img_size, split='train', is_train=True, label_transform=None,
#                  to_tensor=True):
#         super(CDDataset, self).__init__(root_dir, img_size=img_size, split=split, is_train=is_train,
#                                         to_tensor=to_tensor)
#         self.label_transform = label_transform
#
#     def __getitem__(self, index):
#         name = self.img_name_list[index]
#         A_path = get_img_path(self.root_dir, self.img_name_list[index % self.A_size])
#         B_path = get_img_post_path(self.root_dir, self.img_name_list[index % self.A_size])
#         img = np.asarray(Image.open(A_path).convert('RGB'))
#         img_B = np.asarray(Image.open(B_path).convert('RGB'))
#         L_path = get_label_path(self.root_dir, self.img_name_list[index % self.A_size])
#
#         label = np.array(Image.open(L_path), dtype=np.uint8)
#         #  二分类中，前景标注为255
#         if self.label_transform == 'norm':
#             label = label // 255
#
#         [img, img_B], [label] = self.augm.transform([img, img_B], [label], to_tensor=self.to_tensor)
#         # print(label.max())
#         return {'name': name, 'A': img, 'B': img_B, 'L': label}
"""
变化检测数据集
直接扫描文件夹结构：train/A, train/B, train/label 等
"""
import os
from PIL import Image
import numpy as np
import torch
from torch.utils import data
from datasets.data_utils import CDDataAugmentation

IMG_FOLDER_NAME = "A"
IMG_POST_FOLDER_NAME = 'B'
ANNOT_FOLDER_NAME = "label"
IGNORE = 255
label_suffix = '.png'  # 如果标签是 .tif 则改为 '.tif'

def load_img_name_list(root_dir, split):
    img_dir = os.path.join(root_dir, split, IMG_FOLDER_NAME)
    img_name_list = sorted([f for f in os.listdir(img_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))])
    return img_name_list

def get_img_path(root_dir, split, img_name):
    return os.path.join(root_dir, split, IMG_FOLDER_NAME, img_name)

def get_img_post_path(root_dir, split, img_name):
    return os.path.join(root_dir, split, IMG_POST_FOLDER_NAME, img_name)

def get_label_path(root_dir, split, img_name):
    base_name = os.path.splitext(img_name)[0]
    return os.path.join(root_dir, split, ANNOT_FOLDER_NAME, base_name + label_suffix)

class ImageDataset(data.Dataset):
    def __init__(self, root_dir, split='train', img_size=256, is_train=True, to_tensor=True):
        super(ImageDataset, self).__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split
        self.img_name_list = load_img_name_list(root_dir, split)
        self.A_size = len(self.img_name_list)
        self.to_tensor = to_tensor
        if is_train:
            self.augm = CDDataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_vflip=True,
                with_scale_random_crop=True,
                with_random_blur=True,
            )
        else:
            self.augm = CDDataAugmentation(
                img_size=self.img_size
            )

    def __getitem__(self, index):
        name = self.img_name_list[index]
        A_path = get_img_path(self.root_dir, self.split, name)
        B_path = get_img_post_path(self.root_dir, self.split, name)
        img = np.asarray(Image.open(A_path).convert('RGB'))
        img_B = np.asarray(Image.open(B_path).convert('RGB'))
        [img, img_B], _ = self.augm.transform([img, img_B], [], to_tensor=self.to_tensor)
        return {'A': img, 'B': img_B, 'name': name}

    def __len__(self):
        return self.A_size

class CDDataset(ImageDataset):
    def __init__(self, root_dir, img_size, split='train', is_train=True, label_transform=None, to_tensor=True):
        super(CDDataset, self).__init__(root_dir, img_size=img_size, split=split, is_train=is_train, to_tensor=to_tensor)
        self.label_transform = label_transform

    # def __getitem__(self, index):
    #     name = self.img_name_list[index]
    #     A_path = get_img_path(self.root_dir, self.split, name)
    #     B_path = get_img_post_path(self.root_dir, self.split, name)
    #     L_path = get_label_path(self.root_dir, self.split, name)
    #     img = np.asarray(Image.open(A_path).convert('RGB'))
    #     img_B = np.asarray(Image.open(B_path).convert('RGB'))
    #     # 强制灰度读取标签
    #     label = np.array(Image.open(L_path).convert('L'), dtype=np.uint8)
    #     [img, img_B], [label] = self.augm.transform([img, img_B], [label], to_tensor=self.to_tensor)
    #     # 归一化标签到 [0,1]
    #     if label.max() > 1:
    #         label = label.float() / 255.0
    #     # 确保标签形状为 [1, H, W]
    #     if label.dim() == 2:
    #         label = label.unsqueeze(0)
    #     # 确保图像为 float
    #     img = img.float()
    #     img_B = img_B.float()
    #     return {'name': name, 'A': img, 'B': img_B, 'L': label}
    def __getitem__(self, index):
        name = self.img_name_list[index]
        A_path = get_img_path(self.root_dir, self.split, name)
        B_path = get_img_post_path(self.root_dir, self.split, name)
        L_path = get_label_path(self.root_dir, self.split, name)
        img = np.asarray(Image.open(A_path).convert('RGB'))
        img_B = np.asarray(Image.open(B_path).convert('RGB'))
        # 强制灰度读取标签
        label = np.array(Image.open(L_path).convert('L'), dtype=np.uint8)
        [img, img_B], [label] = self.augm.transform([img, img_B], [label], to_tensor=self.to_tensor)

        # 归一化标签到 [0,1]
        if label.max() > 1:
            label = label.float() / 255.0
        else:
            # 如果已经是 0/1 整数，转为浮点
            label = label.float()

        # 确保标签形状为 [1, H, W]
        if label.dim() == 2:
            label = label.unsqueeze(0)

        # 确保图像为 float
        img = img.float()
        img_B = img_B.float()

        # 最终确保标签是浮点（保险）
        label = label.float()

        return {'name': name, 'A': img, 'B': img_B, 'L': label}
