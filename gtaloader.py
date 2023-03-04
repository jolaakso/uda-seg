import torch
import os
import glob
import torchvision as tv

CLASS_COLORS = [
  (0, 0, 0),           # unlabeled
  (0, 0, 0),           # ego vehicle
  (0, 0, 0),           # rectification border
  (0, 0, 0),           # out of roi
  (0, 0, 0),           # static
  (0, 0, 0),           # dynamic
  (0, 0, 0),           # ground
  (128, 64, 128),      # road
  (244, 35, 232),      # sidewalk
  (0, 0, 0),           # parking
  (0, 0, 0),          # rail track
  (70, 70, 70),       # building
  (102, 102, 156),    # wall
  (190, 153, 153),    # fence
  (0, 0, 0),          # guard rail
  (0, 0, 0),          # bridge
  (0, 0, 0),          # tunnel
  (153, 153, 153),    # pole
  (0, 0, 0),          # polegroup
  (250, 170, 30),     # traffic light
  (220, 220,  0),     # traffic sign
  (107, 142, 35),     # vegetation
  (152, 251, 152),    # terrain
  (0, 130, 180),      # sky
  (220, 20, 60),      # person
  (255, 0, 0),        # rider
  (0, 0, 142),        # car
  (0, 0, 70),         # truck
  (0, 60, 100),       # bus
  (0,  0, 0),         # caravan
  (0,  0, 0),         # trailer
  (0, 80, 100),       # train
  (0, 0, 230),        # motorcycle
  (119, 11, 32),      # bicycle
  (0, 0, 0)         # license plate (-1)
]

VOID_CLASSES = torch.Tensor([0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, 34])

VALID_CLASSES = torch.Tensor([
    7,
    8,
    11,
    12,
    13,
    17,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    31,
    32,
    33
])

USED_CLASS_NAMES = torch.Tensor([
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    2,
    0,
    0,
    3,
    4,
    5,
    0,
    0,
    0,
    6,
    0,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    0,
    0,
    17,
    18,
    19,
    0,
]).to(torch.int32)

class GTAVTrainingFileList():
    def __init__(self, path, training_split_ratio=0.9):
        self.path = path
        self.imagespath = os.path.join(path, 'images/')
        self.labelspath = os.path.join(path, 'labels/')

        self.filenames = [os.path.basename(file) for file in glob.glob(f'{self.imagespath}/*.png')]
        training_samples = int(len(self.filenames) * training_split_ratio)
        self.training_names = self.filenames[:training_samples]
        self.validation_names = self.filenames[training_samples:]

    def __getitem__(self, i):
        filename = self.training_names[i]
        return (os.path.join(self.imagespath, filename), os.path.join(self.labelspath, filename))

    def __len__(self):
        return len(self.training_names)


class GTAVValFileList(GTAVTrainingFileList):
    def __init__(self, path, training_split_ratio=0.9):
        super().__init__(path, training_split_ratio)

    def __getitem__(self, i):
        filename = self.validation_names[i]
        return (os.path.join(self.imagespath, filename), os.path.join(self.labelspath, filename))

    def __len__(self):
        return len(self.validation_names)

class CityscapesValFileList():
    def __init__(self, path):
        self.path = path
        self.imagespath = os.path.join(path, 'leftImg8bit/')
        self.labelspath = os.path.join(path, 'gtFine/')
        self.image_names = glob.glob(f'{self.imagespath}/val/*/*.png')
        self.label_names = glob.glob(f'{self.labelspath}/val/*/*_labelIds.png')

    def __getitem__(self, i):
        return (self.image_names[i], self.label_names[i])

    def __len__(self):
        images_len = len(self.image_names)
        assert(images_len == len(self.label_names))
        return images_len

class CityscapesTrainFileList():
    def __init__(self, path):
        self.path = path
        self.imagespath = os.path.join(path, 'leftImg8bit/')
        self.labelspath = os.path.join(path, 'gtFine/')
        self.image_names = glob.glob(f'{self.imagespath}/train/*/*.png')
        self.label_names = glob.glob(f'{self.labelspath}/train/*/*_labelIds.png')

    def __getitem__(self, i):
        return (self.image_names[i], self.label_names[i])

    def __len__(self):
        images_len = len(self.image_names)
        assert(images_len == len(self.label_names))
        return images_len

class TrafficDataset(torch.utils.data.Dataset):
    COLOR_COUNT = 20
    TOTAL_COLOR_COUNT = 35

    def __init__(self, filelist):
        super().__init__()
        self.crop = tv.transforms.RandomCrop((704, 1264))
        self.filelist = filelist

    def __len__(self):
        return len(self.filelist)

    def nullify_voids(self, label_image):
        original_shape = label_image.shape
        return torch.index_select(USED_CLASS_NAMES, 0, label_image.flatten()).reshape(original_shape)

    def __getitem__(self, i):
        image_path, label_path = self.filelist[i]
        image = tv.io.image.read_image(image_path)
        label_image = tv.io.image.read_image(label_path)
        # crop from same place randomly
        image, label_image = torch.split(self.crop(torch.cat((image, label_image), 0)), (3, 1), 0)
        image = tv.transforms.functional.convert_image_dtype(image, dtype=torch.float32)
        label_image = label_image.to(torch.int32)
        label_image = self.nullify_voids(label_image)
        #label_image = tv.transforms.functional.convert_image_dtype(label_image[0], dtype=torch.int64)
        return image, label_image[0].long()
