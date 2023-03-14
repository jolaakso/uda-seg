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
from lsrmodels import DeeplabNW
#import matplotlib.pyplot as plt

BATCH_SIZE = 8
BATCHES_TO_SAVE = 50
EPOCH_COUNT = 125
EPOCH_LENGTH = 1000

class USSegLoss(nn.Module):
    def __init__(self, occupancy=0.3, variability=0.07, saturation=0.05):
        super().__init__()
        self.occupancy = occupancy
        self.variability = variability
        self.saturation = saturation

    def forward(self, source):
        occupancy_score = torch.relu(torch.abs(source.mean() - self.occupancy).mean() - self.saturation)
        flattened_screens = source.flatten(start_dim=2)
        screen_means = flattened_screens.mean(dim=2)
        channel_var_score = torch.relu(torch.abs(screen_means.var(dim=1) - self.variability).mean() - self.saturation)
        dist_from_exponential = torch.relu(torch.abs(flattened_screens.var(dim=2) - screen_means.square()).mean() - self.saturation)

        #vals = torch.tensor([dog, manhattan, channel_population_dist], requires_grad=True, device=self.weights.device, dtype=self.weights.dtype)
        # print(vals * self.weights)
        #print(channel_population_dist)
        if random.randrange(700) == 0:
            print(f'occupancy_score: {occupancy_score}, channel_var_score: {channel_var_score}, dist_from_exponential: {dist_from_exponential}')

        return occupancy_score + channel_var_score + dist_from_exponential

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

def start(save_file_name=None, load_file_name=None, load_backbone=None, load_model=None, dataset_type='gtav', dataset_dir='../datasetit/gtav/', adaptation_dir='../datasetit/cityscapes/', device='cpu', only_adapt=False, unsupervised=False, lock_backbone=False):

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


    epoch_batches = len(dataloader)
    if EPOCH_LENGTH:
        epoch_batches = min(EPOCH_LENGTH, len(dataloader))

    loss_fun = None

    if unsupervised:
        loss_fun = USSegLoss()
    else:
        loss_fun = nn.CrossEntropyLoss(ignore_index=0, weight=loss_weights)

    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=EPOCH_COUNT * len(dataloader), power=0.9)
    epoch = 0

    if load_file_name:
        print(f'Loading checkpoint from file {load_file_name}...')
        checkpoint = torch.load(load_file_name)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        print('Done loading')

    if load_model:
        print(f'Loading model weights separately from {load_model}')
        classifier.load_state_dict(torch.load(load_model)['model_state_dict'])

    if load_backbone:
        print(f'Loading backbone weights from {load_backbone}')
        classifier.backbone.load_state_dict(torch.load(load_backbone)['model_state_dict'])

    if lock_backbone:
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
        train_epoch(dataloader, optimizer, classifier, loss_fun, device, scheduler, unsupervised, lock_backbone)

        if not unsupervised:
            validate(validation_dataloader, classifier, device, pixel_accuracy, mIoU)

        epoch += 1
        if save_file_name:
            print(f'Saving checkpoint to file {save_file_name}...')
            torch.save({
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }, save_file_name)
            print('Done saving')
        print(f'Class priors: {dataset.cropper.label_costs}')

        #if save_file_name:
        #    print(f'Saving model to file {save_file_name}...')
        #    torch.save(classifier.state_dict(), save_file_name)
        #    print('Done saving')

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
    args = parser.parse_args()
    start(args.save_file_name, args.load_file_name, args.load_backbone, args.load_model, args.dataset_type, args.dataset_dir, args.adaptset_dir, args.device, args.only_adapt, args.unsupervised, args.lock_backbone)
