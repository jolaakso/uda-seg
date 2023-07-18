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

CLASS_LABELS = [
    'void',
    'road',
    'sidewalk',
    'building',
    'wall',
    'fence',
    'pole',
    'traffic_light',
    'traffic_sign',
    'vegetation',
    'terrain',
    'sky',
    'person',
    'rider',
    'car',
    'truck',
    'bus',
    'train',
    'motorcycle',
    'bicycle'
]

VALID_CLASS_COLORS = [
  (128, 64, 128),      # road
  (244, 35, 232),      # sidewalk
  (70, 70, 70),       # building
  (102, 102, 156),    # wall
  (190, 153, 153),    # fence
  (153, 153, 153),    # pole
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
  (0, 80, 100),       # train
  (0, 0, 230),        # motorcycle
  (119, 11, 32),      # bicycle
]

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

class CityscapesFileList():
    def __init__(self, path):
        self.path = path
        self.imagespath = os.path.join(path, 'leftImg8bit')
        self.labelspath = os.path.join(path, 'gtFine')
        self.image_names = glob.glob(f'{self.imagespath}/{self.midpath()}/*/*.png')
        # self.label_names = glob.glob(f'{self.labelspath}/{self.midpath()}/*/*_labelIds.png')

    def __getitem__(self, i):
        # leftImg8bit/train/hamburg/hamburg_000000_104857_leftImg8bit.png
        # gtFine/train/hamburg/hamburg_000000_104857_gtFine_labelIds.png
        image_name = self.image_names[i]
        base_name = os.path.basename(image_name)
        city_name = base_name.split('_', 1)[0]
        image_common_part = base_name.rpartition('_')[0]
        label_name = f'{self.labelspath}/{self.midpath()}/{city_name}/{image_common_part}_gtFine_labelIds.png'
        return (image_name, label_name)

    def midpath(self):
        raise NotImplementedError()

    def __len__(self):
        images_len = len(self.image_names)
        return images_len

class CityscapesValFileList(CityscapesFileList):
    def midpath(self):
        return 'val'

class CityscapesTrainFileList(CityscapesFileList):
    def midpath(self):
        return 'train'

class CityscapesTrainExtraFileList(CityscapesTrainFileList):
    def __init__(self, path):
        super().__init__(path)
        train_image_paths = glob.glob(f'{self.imagespath}/{super().midpath()}/*/*.png')
        self.image_names = train_image_paths + self.image_names

    def __getitem__(self, i):
        return (self.image_names[i], None)

    def midpath(self):
        return 'train_extra'

def apply_to_img_and_label(image, label_image, operation):
    mod_image, mod_label_image = torch.split(operation(torch.cat((image, label_image), 0)), (3, 1), 0)
    mod_label_image = mod_label_image.to(label_image.dtype)

    return mod_image, mod_label_image


class UPCropper(torch.nn.Module):
    def __init__(self, crop_size=(720, 1280), samples=1, ignore_label=255, label_count=19, label_costs=None, device='cpu'):
        super().__init__()
        self.crop = tv.transforms.RandomCrop(crop_size)
        self.samples = samples
        self.label_count = label_count
        self.ignore_label = ignore_label
        if label_costs == None:
            self.label_costs = torch.ones(label_count, device=device, dtype=torch.float32) / label_count
        else:
            self.label_costs = label_costs

    def likelihood(self, label_image):
        labels_histogram = torch.bincount(label_image.flatten(), minlength=self.label_count).to(torch.float32)
        if self.ignore_label in range(labels_histogram.numel()):
            labels_histogram[self.ignore_label] = 0
        labels_distribution = torch.nn.functional.normalize(labels_histogram, p=1, dim=0)[:self.label_count]
        normalized_label_costs = torch.nn.functional.normalize(self.label_costs, p=1, dim=0)
        cost = (normalized_label_costs * labels_distribution).sum()

        return cost, labels_distribution

    def crop_randomly(self, image, label_image):
        return apply_to_img_and_label(image, label_image, self.crop)

    def forward(self, image, label_image):
        if self.samples < 1:
            raise RuntimeException(f'samples {self.samples} < 1')
        elif self.samples == 1:
            image, label_image = self.crop_randomly(image, label_image)
            return image, label_image, None

        best_image, best_label_image = self.crop_randomly(image, label_image)
        best_cost, best_distribution = self.likelihood(best_label_image)

        for _ in range(self.samples - 1):
            cand_image, cand_label_image = self.crop_randomly(image, label_image)
            cost, distribution = self.likelihood(cand_label_image.to(label_image.dtype))

            if cost < best_cost:
                best_cost = cost
                best_distribution = distribution
                best_image = cand_image
                best_label_image = cand_label_image

        self.label_costs += best_distribution

        return best_image, best_label_image, best_cost

class ColorRandomizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.white_sampler = torch.distributions.uniform.Uniform(0.77, 1.417)
        self.shift_sampler = torch.distributions.uniform.Uniform(-0.1, 0.1)

    def forward(self, x):
        new_white = self.white_sampler.sample((3, 1, 1)).to(x.device).to(x.dtype)
        color_shift = self.shift_sampler.sample((3, 1, 1)).to(x.device).to(x.dtype)
        return x * new_white + color_shift

class TrafficDataset(torch.utils.data.Dataset):
    COLOR_COUNT = 19
    TOTAL_COLOR_COUNT = 35

    def __init__(self, filelist, resize=(720, 1280), cropper=None, train_augmentations=False, device='cpu', include_label_cost=False, allow_missing_labels=False):
        super().__init__()
        self.device = device
        self.img_resize = tv.transforms.Resize(resize, interpolation=tv.transforms.InterpolationMode.BICUBIC)
        self.label_resize = tv.transforms.Resize(resize, interpolation=tv.transforms.InterpolationMode.NEAREST)
        self.normalize_colors = tv.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.filelist = filelist
        self.train_augmentations = train_augmentations
        if train_augmentations:
            self.augmentations = tv.transforms.Compose([
                tv.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                ColorRandomizer(),
                tv.transforms.GaussianBlur(5, sigma=(0.01, 1.0))
            ])

            self.common_augs = tv.transforms.RandomHorizontalFlip()
        self.cropper = None
        if cropper:
            self.cropper = cropper
        self.used_class_names = USED_CLASS_NAMES.to(device)
        self.include_label_cost = include_label_cost
        self.allow_missing_labels = allow_missing_labels

    def __len__(self):
        return len(self.filelist)

    def nullify_voids(self, label_image):
        original_shape = label_image.shape
        nulled = torch.index_select(self.used_class_names, 0, label_image.flatten()).reshape(original_shape)
        # -1 as void class
        nulled = nulled - 1
        # 255 as void class
        nulled = torch.where(nulled == -1, 255, nulled)
        return nulled

    def __getitem__(self, i):
        image_path, label_path = self.filelist[i]
        image = tv.io.image.read_image(image_path, tv.io.ImageReadMode.RGB).to(self.device)
        label_image = None
        if label_path != None:
            label_image = tv.io.image.read_image(label_path).to(self.device)
        elif self.allow_missing_labels:
            label_image = torch.zeros_like(image)[0, :, :].unsqueeze(0)

        if not self.allow_missing_labels and label_image == None:
            raise ValueError(f'label_image is None, but allow_missing_labels is set to {self.allow_missing_labels}')
        # crop from same place randomly
        image = self.img_resize(image)
        label_image = self.label_resize(label_image)
        image = tv.transforms.functional.convert_image_dtype(image, dtype=torch.float32)
        label_image = label_image.to(torch.int32)
        label_image = self.nullify_voids(label_image)
        cost = 0
        if self.cropper:
            image, label_image, cost = self.cropper(image, label_image)
        if self.train_augmentations:
            image = self.augmentations(image)
            image, label_image = apply_to_img_and_label(image, label_image, self.common_augs)
        image = self.normalize_colors(image)

        if self.cropper and self.include_label_cost:
            return image, label_image[0].long(), cost
        #label_image = tv.transforms.functional.convert_image_dtype(label_image[0], dtype=torch.int64)
        return image, label_image[0].long()
