import torch
import torchvision as tv
from torchmetrics.classification import MulticlassJaccardIndex
from torch import nn
from gtaloader import TrafficDataset, GTAVTrainingFileList, GTAVValFileList, CityscapesValFileList, CityscapesTrainFileList, UPCropper
import argparse
import sys
from acc_conv import AccConv2d
from dcanetv2 import DCANetV2
from BigConv import BigConvBlock
import random
import math
from lsrmodels import DeeplabNW
import FeatureModifier as fn
#import matplotlib.pyplot as plt

BATCH_SIZE = 8
LEARNING_RATE = 0.00025
BATCHES_TO_SAVE = 50
EPOCH_COUNT = 70
EPOCH_LENGTH = 1000

class TeacherLoss(nn.Module):
    def __init__(self, teacher):
        super().__init__()
        teacher.eval()
        self.teacher = teacher

    def forward(self, x):
        return self.teacher(x).mean()

# Based on torchvision/models/segmentation/_utils.py
class TeacherTrainerWrapper(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier

    def forward(self, x, input_size):
        x = self.classifier(x)
        x = torch.nn.functional.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        return { 'out': x }

class FeatureWeighter(nn.Module):
    def __init__(self, class_frequencies):
        super().__init__()

    def forward(self, x, class_map):
        return x


class SelfOrganizingNorm2d(nn.Module):
    # All coords in unit sphere with learnable radius?
    def __init__(self, channels, proximity_dims=1, p=2, eps=1e-12):
        super().__init__()
        self.normalization_space = nn.Parameter(torch.randn(channels, proximity_dims))
        self.p = p
        self.gamma = nn.Parameter(torch.randn(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.randn(1, channels, 1, 1))
        self.eps = eps

    def _weighted_mean2d(self, x, weights, weights_sum):
        x = x.mean(3).mean(2)
        x = nn.functional.linear(x, weights) / weights_sum

        return x.unsqueeze(2).unsqueeze(2)

    def forward(self, x):
        # This and sum of proximities should be precomputed when in eval mode
        proximities = torch.cdist(self.normalization_space, self.normalization_space, p=self.p)
        proximities = torch.exp(-proximities)
        proximities_sum = proximities.sum(1)

        w_means = self._weighted_mean2d(x, proximities, proximities_sum)

        mean_shifted = x - w_means

        w_vars = self._weighted_mean2d(mean_shifted.square(), proximities, proximities_sum)

        x = mean_shifted / (w_vars + self.eps).sqrt()

        return x * self.gamma + self.beta

def append_sonorm_to_rn50(backbone):
    backbone.get_submodule('layer1.1').add_module('self_organizing_norm', SelfOrganizingNorm2d(256, 2))
    backbone.get_submodule('layer1.2').add_module('self_organizing_norm', SelfOrganizingNorm2d(256, 2))
    backbone.get_submodule('layer2.1').add_module('self_organizing_norm', SelfOrganizingNorm2d(512, 2))
    backbone.get_submodule('layer2.2').add_module('self_organizing_norm', SelfOrganizingNorm2d(512, 2))
    backbone.get_submodule('layer2.3').add_module('self_organizing_norm', SelfOrganizingNorm2d(512, 2))
    backbone.get_submodule('layer3.1').add_module('self_organizing_norm', SelfOrganizingNorm2d(1024, 2))
    backbone.get_submodule('layer3.2').add_module('self_organizing_norm', SelfOrganizingNorm2d(1024, 2))
    backbone.get_submodule('layer3.3').add_module('self_organizing_norm', SelfOrganizingNorm2d(1024, 2))
    backbone.get_submodule('layer3.4').add_module('self_organizing_norm', SelfOrganizingNorm2d(1024, 2))
    backbone.get_submodule('layer3.5').add_module('self_organizing_norm', SelfOrganizingNorm2d(1024, 2))
    backbone.get_submodule('layer4.1').add_module('self_organizing_norm', SelfOrganizingNorm2d(2048, 2))
    backbone.get_submodule('layer4.2').add_module('self_organizing_norm', SelfOrganizingNorm2d(2048, 2))

    print(backbone)

class USSegLoss(nn.Module):
    def __init__(self, occupancy=0.21, variability=0.0181, saturation=0.05):
        super().__init__()
        self.occupancy = occupancy
        self.variability = variability
        self.saturation = saturation
        self.eps = 1e-12

    def diff_ent(self, source):
        H_y = 1.4189       # (1 + math.log(2 * math.pi)) / 2
        k_1 = 7.4129       # 36 / (8 * math.sqrt(3) - 9)
        k_2 = 33.6694      # 24 / (16 * math.sqrt(3) - 27)
        half_sqrt = 0.7071 # math.sqrt(1/2)
        x = source
        asymmetry_term = (x * (-x.square() / 2).exp()).mean().square()
        sparsity_term = ((-x.square() / 2).exp().mean() - half_sqrt).square()
        return H_y - k_1 * asymmetry_term + k_2 * sparsity_term

    def non_gauss_score(self, source):
        occupancy_score = torch.square(source.mean() - self.occupancy).mean()
        flattened_screens = source.flatten(start_dim=2)
        screen_means = flattened_screens.mean(dim=2)
        channel_var_score = torch.square(screen_means.var(dim=1) - self.variability).mean()
        diff_ent_score = self.diff_ent(source)

        if random.randrange(700) == 0:
            print(f'occupancy_score: {occupancy_score}, channel_var_score: {channel_var_score}, diff_ent: {diff_ent_score}')

        return occupancy_score + channel_var_score + diff_ent_score

    def sq_stat_score(self, source):
        occupancy_score = torch.square(source.mean() - self.occupancy).mean()
        flattened_screens = source.flatten(start_dim=2)
        screen_means = flattened_screens.mean(dim=2)
        channel_var_score = torch.square(screen_means.var(dim=1) - self.variability).mean()
        dist_from_exponential = torch.square(flattened_screens.var(dim=2) - screen_means.square()).mean()

        #vals = torch.tensor([dog, manhattan, channel_population_dist], requires_grad=True, device=self.weights.device, dtype=self.weights.dtype)
        # print(vals * self.weights)
        #print(channel_population_dist)
        if random.randrange(700) == 0:
            print(f'occupancy_score: {occupancy_score}, channel_var_score: {channel_var_score}, dist_from_exponential: {dist_from_exponential}')

        return occupancy_score + channel_var_score + dist_from_exponential

    def abs_stat_score(self, source):
        occupancy_score = torch.relu((torch.abs(source.mean() - self.occupancy) / self.occupancy).mean() - self.saturation)
        flattened_screens = source.flatten(start_dim=2)
        screen_means = flattened_screens.mean(dim=2)
        channel_var_score = torch.relu((torch.abs(screen_means.var(dim=1) - self.variability) / self.variability).mean()/self.variability - self.saturation)
        square_screen_means = screen_means.square()
        dist_from_exponential = torch.relu((torch.abs(flattened_screens.var(dim=2) - square_screen_means)/(square_screen_means + self.eps)).mean() - self.saturation)

        #vals = torch.tensor([dog, manhattan, channel_population_dist], requires_grad=True, device=self.weights.device, dtype=self.weights.dtype)
        # print(vals * self.weights)
        #print(channel_population_dist)
        if random.randrange(700) == 0:
            print(f'occupancy_score: {occupancy_score}, channel_var_score: {channel_var_score}, dist_from_exponential: {dist_from_exponential}')

        return occupancy_score + channel_var_score + dist_from_exponential

    def forward(self, source):
        return self.sq_stat_score(source)


def pixel_accuracy(predictions, batch_labels):
    # Assumes softmax input images
    predicted_labels = predictions.argmax(dim=1)
    correct = torch.eq(predicted_labels, batch_labels).int()
    return float(correct.sum()) / float(correct.numel())

def train_epoch(dataloader, optimizer, classifier, loss_fun, device, scheduler, unsupervised=False, lock_backbone=False):
    batch_count = 0
    classifier.train()
    if lock_backbone:
        classifier.backbone.eval()
    total_loss = 0
    for ix, (batch_images, batch_labels) in enumerate(dataloader):
        optimizer.zero_grad()
        batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
        predictions = classifier(batch_images)['out']
        loss = None
        if unsupervised:
            loss = loss_fun(torch.tanh(predictions))
        else:
            loss = loss_fun(predictions, batch_labels)
        #print(list(list(classifier.children())[-1][0].children())[0].weight.is_leaf)
        loss.backward()
        optimizer.step()
        if batch_count % 10 == 0:
            print(f'Training batch: {batch_count}, Loss: {loss.item()}, learning rate: {scheduler.get_last_lr()}')
            #plt.imshow(predictions[0][0].detach().numpy())
            #plt.show()
        total_loss += loss.item()
        batch_count += 1
        scheduler.step()
        if EPOCH_LENGTH and batch_count >= EPOCH_LENGTH:
            break
    total_loss /= float(batch_count)
    print(f'Mean training loss for epoch: {total_loss}')

def train_teacher(dataloader, optimizer, backbone, dumb_backbone, head, teacher, head_loss_fun, teacher_loss_fun, device, scheduler):
    print('Starting teacher training.')
    batch_count = 0
    teacher.train()
    head.train()
    backbone.eval()
    dumb_backbone.eval()

    split_size = dataloader.batch_size // 2

    total_loss = 0
    for ix, (batch_images, batch_labels) in enumerate(dataloader):
        optimizer.zero_grad()
        batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
        trained_images, dumb_images = batch_images.split(split_size, dim=0)
        features = torch.cat((backbone(trained_images)['out'], dumb_backbone(dumb_images)['out']), dim=0)
        features.requires_grad = True
        head_loss_fun(head(features, batch_images.shape[-2:])['out'], batch_labels).backward()

        target_grad = features.grad

        teacher_loss = teacher_loss_fun(teacher(features), target_grad)
        teacher_loss.backward()

        # zero to make sure no update on these
        head.zero_grad()
        backbone.zero_grad()
        dumb_backbone.zero_grad()
        optimizer.step()

        if batch_count % 10 == 0:
            print(f'Training batch: {batch_count}, Teacher loss: {teacher_loss.item()}, learning rate: {scheduler.get_last_lr()}')
        total_loss += teacher_loss.item()
        batch_count += 1
        scheduler.step()
        if EPOCH_LENGTH and batch_count >= EPOCH_LENGTH:
            break
    total_loss /= float(batch_count)
    print(f'Mean teacher loss for epoch: {total_loss}')

def student_train(dataloader, optimizer, backbone, loss_fun, device, scheduler):
    print('Starting student training using teacher')
    batch_count = 0
    backbone.train()

    split_size = dataloader.batch_size // 2

    total_loss = 0
    for ix, (batch_images, _) in enumerate(dataloader):
        optimizer.zero_grad()
        batch_images = batch_images.to(device)
        features = backbone(batch_images)['out']

        loss = loss_fun(features)
        loss.backward()

        optimizer.step()

        if batch_count % 10 == 0:
            print(f'Training batch: {batch_count}, loss: {loss.item()}, learning rate: {scheduler.get_last_lr()}')
        total_loss += loss.item()
        batch_count += 1
        scheduler.step()
        if EPOCH_LENGTH and batch_count >= EPOCH_LENGTH:
            break
    total_loss /= float(batch_count)
    print(f'Mean student loss for epoch: {total_loss}')

def validate(dataloader, classifier, device, validation_loss_fun, mIoUGainFun):
    classifier.eval()
    print('Validating...')
    batch_count = 0
    total_batches = len(dataloader)
    validation_loss = 0
    total_mIoU_gain = 0
    total_classIoU_gain = None
    with torch.no_grad():
        for batch_images, batch_labels in dataloader:
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            predictions = classifier(batch_images)['out']
            normalized_masks = predictions.softmax(dim=1)
            loss = validation_loss_fun(normalized_masks, batch_labels)
            mIoUGainFun.average = 'macro'
            mIoUGain = mIoUGainFun(normalized_masks, batch_labels)
            mIoUGainFun.average = 'none'
            classIoUGain = mIoUGainFun(normalized_masks, batch_labels)
            validation_loss += loss
            total_mIoU_gain += mIoUGain

            if total_classIoU_gain == None:
                total_classIoU_gain = classIoUGain
            else:
                total_classIoU_gain += classIoUGain

            if batch_count % 10 == 0:
                print(f'Validation batch: {batch_count}, Loss: {loss}, mIoU: {mIoUGain}')
            batch_count += 1
    validation_loss /= float(total_batches)
    total_mIoU_gain /= float(total_batches)
    total_classIoU_gain /= float(total_batches)
    print(f'Validation error: {validation_loss}')
    print(f'IoU error: {total_mIoU_gain}')
    print(f'Class IoU errors: {total_classIoU_gain}')

def load_gtav_set(dataset_dir, device='cpu'):
    filelist = GTAVTrainingFileList(dataset_dir, training_split_ratio=0.95)
    val_filelist = GTAVValFileList(dataset_dir, training_split_ratio=0.97)
    assert len(set(filelist) & set(val_filelist)) == 0
    # orig size (704, 1264)?
    up_cropper = UPCropper(device=device, crop_size=(512, 512), samples=5)

    dataset = TrafficDataset(filelist, resize=(720, 1280), train_augmentations=True, cropper=up_cropper, device=device)
    val_dataset = TrafficDataset(val_filelist, resize=(720, 1280), device=device)

    return dataset, val_dataset

def load_cityscapes_set(dataset_dir, device='cpu'):
    filelist = CityscapesTrainFileList(dataset_dir)
    val_filelist = CityscapesValFileList(dataset_dir)
    assert len(set(filelist) & set(val_filelist)) == 0

    up_cropper = UPCropper(device=device, crop_size=(512, 1024), samples=1)

    dataset = TrafficDataset(filelist, resize=(512, 1024), train_augmentations=True, cropper=up_cropper, device=device)
    val_dataset = TrafficDataset(val_filelist, resize=(512, 1024), device=device)
    #val_dataset = TrafficDataset(val_filelist, resize=(512, 1024), crop_size=(512, 1024))

    return dataset, val_dataset

def start(save_file_name=None, load_file_name=None, load_backbone=None, load_model=None, dataset_type='gtav', dataset_dir='../datasetit/gtav/', adaptation_dir='../datasetit/cityscapes/', device='cpu', only_adapt=False, unsupervised=False, lock_backbone=False, teacher_mode=False, student_mode=False, load_teacher=None):

    dataset, val_dataset = (None, None)
    loss_weights = None
    if dataset_type == 'gtav':
        dataset, val_dataset = load_gtav_set(dataset_dir, device=device)
        #loss_weights = torch.Tensor([0.0, 2.7346e+00, 1.0929e+01, 5.3414e+00, 4.7041e+01, 1.3804e+02,
        #                             8.5594e+01, 6.6778e+02, 1.0664e+03, 1.1754e+01, 4.1259e+01, 6.5842e+00,
        #                             2.4710e+02, 3.0451e+03, 3.4462e+01, 7.5414e+01, 2.4990e+02, 1.4041e+03,
        #                             2.7946e+03, 1.7960e+04]).to(device)
        print(f'Loaded GTAV dataset at {dataset_dir}')
    elif dataset_type == 'cityscapes':
        dataset, val_dataset = load_cityscapes_set(dataset_dir)
        print(f'Loaded Cityscapes dataset at {dataset_dir}')
    else:
        raise Exception(f'Unknown dataset type {dataset}')

    dataloader = torch.utils.data.DataLoader(dataset, drop_last=True, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(val_dataset, drop_last=True, batch_size=BATCH_SIZE)
    mIoU = MulticlassJaccardIndex(num_classes = dataset.COLOR_COUNT, average = 'macro', ignore_index=0).to(device)
    print('Dataloader initialized')
    # params 11029075 (mobilenetv3)
    # params 10413651 (DCANet)
    classifier = tv.models.segmentation.deeplabv3_resnet50(num_classes = dataset.COLOR_COUNT)
    # classifier = DeeplabNW(num_classes = dataset.COLOR_COUNT, backbone='resnet50', pretrained=False)

    teacher = None
    untrained_backbone = None
    if teacher_mode:
        print('Initializing teacher network')
        untrained_backbone = tv.models.segmentation.deeplabv3_resnet50(num_classes = 2048).backbone
        teacher = fn.FeatureModifier()

        for p in untrained_backbone.parameters():
            p.requires_grad = False

        teacher = teacher.to(device)
        untrained_backbone = untrained_backbone.to(device)

    if student_mode:
        print('Initalizing teacher to train in student mode')
        teacher = fn.FeatureModifier()

        if not load_teacher:
            raise RuntimeError('--load-teacher needs to be defined in student mode')

        print(f'Loading teacher from {load_teacher}')
        teacher.load_state_dict(torch.load(load_teacher)['teacher_state_dict'])

        for p in teacher.parameters():
            p.requires_grad = False

        teacher = teacher.to(device)


    if unsupervised:
        classifier = classifier.backbone
    #classifier.classifier[0] = nn.Sequential(
    #    nn.Conv2d(2048, 128, 1),
    #    nn.ReLU(),
    #    BigConvBlock(88, 158, 128),
    #    nn.ReLU(),
    #    nn.Conv2d(128, 2048, 1),
    #    nn.ReLU(),
    #    nn.BatchNorm2d(2048),
    #    classifier.classifier[0])
    classifier = classifier.to(device)

    #append_sonorm_to_rn50(classifier.backbone)

    epoch_batches = len(dataloader)
    if EPOCH_LENGTH:
        epoch_batches = min(EPOCH_LENGTH, len(dataloader))

    loss_fun = None
    optim_params = []

    if unsupervised:
        loss_fun = USSegLoss()
        optim_params = [{'params': classifier.parameters(), 'lr': LEARNING_RATE }]
    elif lock_backbone:
        loss_fun = nn.CrossEntropyLoss(ignore_index=0, weight=loss_weights)
        optim_params = [{'params': classifier.backbone.parameters(), 'lr': 0 },
                        { 'params': classifier.classifier.parameters(), 'lr': 10 * LEARNING_RATE }]
    elif teacher_mode:
        loss_fun = nn.CrossEntropyLoss(ignore_index=0, weight=loss_weights)
        optim_params = [{'params': teacher.parameters(), 'lr': 10 * LEARNING_RATE}]
    elif student_mode:
        loss_fun = TeacherLoss(teacher)
        optim_params = [{ 'params': classifier.backbone.parameters(), 'lr': LEARNING_RATE }]
    else:
        loss_fun = nn.CrossEntropyLoss(ignore_index=0, weight=loss_weights)
        optim_params = [{'params': classifier.backbone.parameters(), 'lr': LEARNING_RATE },
                        { 'params': classifier.classifier.parameters(), 'lr': 10 * LEARNING_RATE }]

    # 10x LR for classifier, 1x for backbone
    optimizer = torch.optim.SGD(optim_params, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=EPOCH_COUNT * len(dataloader), power=0.9)
    epoch = 0

    if load_file_name:
        print(f'Loading checkpoint from file {load_file_name}...')
        checkpoint = torch.load(load_file_name)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        if teacher_mode and 'teacher_state_dict' in checkpoint:
            teacher.load_state_dict(checkpoint['teacher'])
        print('Done loading')

    if load_model:
        print(f'Loading model weights separately from {load_model}')
        classifier.load_state_dict(torch.load(load_model)['model_state_dict'])

    if load_backbone:
        print(f'Loading backbone weights from {load_backbone}')
        classifier.backbone.load_state_dict(torch.load(load_backbone)['model_state_dict'])

    if lock_backbone:
        print('Adding Tanh layer')
        classifier.backbone.add_module('tanh', nn.Tanh())
        print(f'Locking model backbone')
        for p in classifier.backbone.parameters():
            p.requires_grad = False

    #classifier.backbone.load_state_dict(torch.load('/wrk/users/jola/results/unsupervised-exp-dist-metric-abs-saturation-005-only-target.torch'))
    #for p in classifier.backbone.parameters():
    #    p.requires_grad = False
    #print('Loaded backbone dict')

    print('Network initialized')

    print('Starting optimization')

    if only_adapt:
        adaptation_filelist = CityscapesValFileList(adaptation_dir)
        adaptation_dataloader = torch.utils.data.DataLoader(TrafficDataset(adaptation_filelist), drop_last=True, batch_size=BATCH_SIZE)
        print('== DA Validation ==')
        validate(adaptation_dataloader, classifier, device, pixel_accuracy, mIoU)
        print('== DA Validation done ==')
        return
    if save_file_name:
        print(f'Will save the checkpoint every epoch to {save_file_name}')
    while epoch < EPOCH_COUNT:
        print(f'Epoch: {epoch}')
        if teacher_mode:
            # (dataloader, optimizer, backbone, dumb_backbone, head, teacher, head_loss_fun, teacher_loss_fun, device, scheduler)
            backbone = classifier.backbone
            for p in backbone.parameters():
                p.requires_grad = False
            head = TeacherTrainerWrapper(classifier.classifier)
            train_teacher(dataloader, optimizer, backbone, untrained_backbone, head, teacher, loss_fun, nn.MSELoss(), device, scheduler)
        elif student_mode:
            backbone = classifier.backbone
            for p in classifier.classifier.parameters():
                p.requires_grad = False
            student_train(dataloader, optimizer, backbone, loss_fun, device, scheduler)
        else:
            train_epoch(dataloader, optimizer, classifier, loss_fun, device, scheduler, unsupervised, lock_backbone)

        if not unsupervised and not teacher_mode:
            validate(validation_dataloader, classifier, device, pixel_accuracy, mIoU)

        epoch += 1
        if save_file_name:
            print(f'Saving checkpoint to file {save_file_name}...')
            save_dict = {
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }

            if teacher_mode:
                save_dict['teacher_state_dict'] = teacher.state_dict()
            else:
                save_dict['model_state_dict'] = classifier.state_dict()

            torch.save(save_dict, save_file_name)
            print('Done saving')
        print(f'Class priors: {dataset.cropper.label_costs}')

        #if save_file_name:
        #    print(f'Saving model to file {save_file_name}...')
        #    torch.save(classifier.state_dict(), save_file_name)
        #    print('Done saving')

    print('Finished optimization')

if __name__ == "__main__":
    #img = tv.io.read_image('english-black-lab-puppy.jpg')
    #l = USSegLoss()
    #l(img)
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--save', dest='save_file_name')
    parser.add_argument('--load', dest='load_file_name')
    parser.add_argument('--dataset', dest='dataset_dir')
    parser.add_argument('--dataset-type', dest='dataset_type')
    parser.add_argument('--adaptset', dest='adaptset_dir')
    parser.add_argument('--device', dest='device')
    parser.add_argument('--load-backbone', dest='load_backbone')
    parser.add_argument('--load-model', dest='load_model')
    parser.add_argument('--unsupervised', dest='unsupervised', action='store_true')
    parser.add_argument('--lock-backbone', dest='lock_backbone', action='store_true')
    parser.add_argument('--only-adapt', dest='only_adapt', action='store_true')
    parser.add_argument('--teacher-mode', dest='teacher_mode', action='store_true')
    parser.add_argument('--student-mode', dest='student_mode', action='store_true')
    parser.add_argument('--load-teacher', dest='load_teacher')
    args = parser.parse_args()
    start(args.save_file_name, args.load_file_name, args.load_backbone, args.load_model, args.dataset_type, args.dataset_dir, args.adaptset_dir, args.device, args.only_adapt, args.unsupervised, args.lock_backbone, args.teacher_mode, args.student_mode, args.load_teacher)
