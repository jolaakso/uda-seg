import torch
import torchvision as tv
from torchmetrics.classification import MulticlassJaccardIndex
from torch import nn
from gtaloader import TrafficDataset, GTAVTrainingFileList, GTAVValFileList, CityscapesValFileList
import argparse
import sys
from acc_conv import AccConv2d
from dcanetv2 import DCANetV2
from BigConv import BigConvBlock
import random
#import matplotlib.pyplot as plt

BATCH_SIZE = 2
BATCHES_TO_SAVE = 50
EPOCH_COUNT = 7

class USSegLoss(nn.Module):
    def __init__(self, weights=torch.Tensor([2000.0, 1.0, 1.0]).to(torch.float).to('cpu'), overlap_ratio=0.1):
        super().__init__()
        self.wide_gaussian = tv.transforms.GaussianBlur(7, 5)
        self.narrow_gaussian = tv.transforms.GaussianBlur(7, 2)
        self.weights = weights
        self.overlap_ratio = overlap_ratio
        self.split = 16
        self.occupancy = 0.3
        self.variability = 0.07
        self.saturation = 0.007

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

def train_epoch(dataloader, optimizer, classifier, loss_fun, device):
    batch_count = 1
    classifier.train()
    total_loss = 0
    for ix, (batch_images, batch_labels) in enumerate(dataloader):
        optimizer.zero_grad()
        batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
        predictions = classifier(batch_images)['out']
        #normalized_masks = predictions.softmax(dim=1)
        loss = loss_fun(torch.tanh(predictions))
        #print(list(list(classifier.children())[-1][0].children())[0].weight.is_leaf)
        loss.backward()
        optimizer.step()
        if batch_count % 10 == 0:
            print(f'Training batch: {batch_count}, Loss: {loss.item()}')
            #plt.imshow(predictions[0][0].detach().numpy())
            #plt.show()
        total_loss += loss.item()
        batch_count += 1
    total_loss /= float(batch_count)
    print(f'Mean training loss for epoch: {total_loss}')

def validate(dataloader, classifier, device, validation_loss_fun, mIoUGainFun):
    classifier.eval()
    print('Validating...')
    batch_count = 1
    total_batches = len(dataloader)
    validation_loss = 0
    total_mIoU_gain = 0
    with torch.no_grad():
        for batch_images, batch_labels in dataloader:
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            predictions = classifier(batch_images)['out']
            normalized_masks = predictions.softmax(dim=1)
            loss = validation_loss_fun(normalized_masks, batch_labels)
            mIoUGain = mIoUGainFun(normalized_masks, batch_labels)
            validation_loss += loss
            total_mIoU_gain += mIoUGain
            if batch_count % 10 == 0:
                print(f'Validation batch: {batch_count}, Loss: {loss}, mIoU: {mIoUGain}')
            batch_count += 1
    validation_loss /= float(total_batches)
    total_mIoU_gain /= float(total_batches)
    print(f'Validation error: {validation_loss}')
    print(f'IoU error: {total_mIoU_gain}')

def start(save_file_name=None, load_file_name=None, dataset_dir='../datasetit/gtav/', adaptation_dir='../datasetit/cityscapes/', device='cpu', only_adapt=False):
    filelist = GTAVTrainingFileList(dataset_dir)
    val_filelist = GTAVValFileList(dataset_dir)
    dataset = TrafficDataset(filelist)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(TrafficDataset(val_filelist), batch_size=BATCH_SIZE)
    mIoU = MulticlassJaccardIndex(num_classes = dataset.COLOR_COUNT, average = 'macro').to(device)
    print('Dataloader initialized')
    # params 11029075 (mobilenetv3)
    # params 10413651 (DCANet)
    classifier = tv.models.segmentation.deeplabv3_resnet50(num_classes = dataset.COLOR_COUNT).backbone
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

    if load_file_name:
        print(f'Loading model from file {load_file_name}...')
        classifier.load_state_dict(torch.load(load_file_name))
        print('Done loading')

    print('Network initialized')
    loss_fun = USSegLoss(weights=torch.Tensor([2000.0, 1.0, 1.0]).to(torch.float).to(device), overlap_ratio=0.33)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.9)
    print('Starting optimization')
    if only_adapt:
        adaptation_filelist = CityscapesValFileList(adaptation_dir)
        adaptation_dataloader = torch.utils.data.DataLoader(TrafficDataset(adaptation_filelist), batch_size=BATCH_SIZE)
        print('== DA Validation ==')
        validate(adaptation_dataloader, classifier, device, pixel_accuracy, mIoU)
        print('== DA Validation done ==')
        return
    if save_file_name:
        print(f'Will save the model every epoch to {save_file_name}')
    for epoch in range(EPOCH_COUNT):
        print(f'Epoch: {epoch}')
        train_epoch(dataloader, optimizer, classifier, loss_fun, device)
        #validate(validation_dataloader, classifier, device, pixel_accuracy, mIoU)
        if save_file_name:
            print(f'Saving model to file {save_file_name}...')
            torch.save(classifier.state_dict(), save_file_name)
            print('Done saving')
        scheduler.step()

if __name__ == "__main__":
    #img = tv.io.read_image('english-black-lab-puppy.jpg')
    #l = USSegLoss()
    #l(img)
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--save', dest='save_file_name')
    parser.add_argument('--load', dest='load_file_name')
    parser.add_argument('--dataset', dest='dataset_dir')
    parser.add_argument('--adaptset', dest='adaptset_dir')
    parser.add_argument('--device', dest='device')
    parser.add_argument('--only-adapt', dest='only_adapt', action='store_true')
    args = parser.parse_args()
    start(args.save_file_name, args.load_file_name, args.dataset_dir, args.adaptset_dir, args.device, args.only_adapt)
