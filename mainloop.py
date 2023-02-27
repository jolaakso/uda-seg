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
EPOCH_COUNT = 10

class USSegLoss(nn.Module):
    def __init__(self, weights=torch.Tensor([2000.0, 1.0, 1.0]).to(torch.float).to('cpu'), overlap_ratio=0.1):
        super().__init__()
        self.wide_gaussian = tv.transforms.GaussianBlur(7, 5)
        self.narrow_gaussian = tv.transforms.GaussianBlur(7, 2)
        self.weights = weights
        self.overlap_ratio = overlap_ratio
        self.split = 16

    def forward(self, source):
        target_populated_channels = self.overlap_ratio / float(source.shape[1])

        v_masks = torch.ones_like(source)
        h_masks = v_masks.sum(dim=2, keepdim=True) / float(source.shape[3] * source.shape[2])
        h_masks = 2*h_masks.cumsum(dim=3) - 1
        v_masks = v_masks.sum(dim=3, keepdim=True) / float(source.shape[3] * source.shape[2])
        v_masks = 2*v_masks.cumsum(dim=2) - 1
        #print(v_masks.shape)
        #print(h_masks.shape)
        v_dist = torch.softmax(source.sum(dim=3, keepdim=True), dim=2)
        h_dist = torch.softmax(source.sum(dim=2, keepdim=True), dim=3)
        v_vars = torch.sqrt(torch.var((v_dist * v_masks), dim=2))
        h_vars = torch.sqrt(torch.var((h_dist * h_masks), dim=3))

        #chunked?
        spreads = v_vars.mean() + h_vars.mean()

        sparcity = -torch.square(torch.square(source - 0.5).mean())

        #loc_means = torch.cat(((v_dist * v_masks).mean(dim=2), (h_dist * h_masks).mean(dim=3)), 2)
        #loc_chunks = torch.chunk(loc_means, 16, dim=1)
        #loc_vars = []
        #for c in loc_chunks:
            #print(torch.flatten(c.mean(dim=1), start_dim=1).shape)
        #    loc_vars.append(torch.var(c.mean(dim=2), dim=1, keepdim=True))
            #print(torch.flatten(torch.prod(c, dim=1, keepdim=True), start_dim=1).mean(dim=1, keepdim=True).shape)
            #print(c[0][0:20])

        #loc_variance = -torch.cat(loc_vars, 1).mean()

        # Non-smoothness
        #dog = nn.functional.relu(self.wide_gaussian(source) - self.narrow_gaussian(source))
        #dog = torch.square(dog.mean())
        #dog = 0

        #absolutes = torch.abs(source)

        # L1
        #manhattan = absolutes.mean()
        #manhattan = 0

        # Deviance_from_target
        #mean_channels = source.mean(dim=3).mean(dim=2)
        #deviance = torch.square(mean_channels - (torch.ones_like(mean_channels) * self.overlap_ratio)).mean()

        #expected_vals = nn.functional.softmax(torch.flatten(source, start_dim=2), dim=2) * torch.flatten(loc_masks, start_dim=2)
        #expected_vals = expected_vals.sum(dim=2, keepdim=True)

        channel_population = source.mean(dim=3).mean(dim=2)
        limits = torch.ones_like(channel_population) / float(channel_population.shape[1])
        limits = (limits.cumsum(dim=1) * 0.4) + 0.05
        #print(channel_population.shape)
        #print(limits[0][0])
        #print(limits[0][-1])
        #print(limits.shape)
        #print(channel_population[0][0:100] - self.overlap_ratio)
        #print(torch.flatten(source.mean(dim=1), start_dim=1).shape)
        channel_population_dist = torch.square(channel_population - limits).mean()
        chunks = torch.chunk(source, self.split, dim=1)
        vars = []
        for c in chunks:
            #print(torch.flatten(c.mean(dim=1), start_dim=1).shape)
            vars.append(torch.var(torch.flatten(c.mean(dim=1), start_dim=1), dim=1, keepdim=True))
            #print(torch.flatten(torch.prod(c, dim=1, keepdim=True), start_dim=1).mean(dim=1, keepdim=True).shape)
            #print(c[0][0:20])
        #print(vars[0].shape)
        variance = torch.cat(vars, 1).mean()
        #print(variance)
        #normalizeds = nonnegatives / maxs

        #vals = torch.tensor([dog, manhattan, channel_population_dist], requires_grad=True, device=self.weights.device, dtype=self.weights.dtype)
        # print(vals * self.weights)
        #print(channel_population_dist)
        if random.randrange(700) == 0:
            print(f'dist: {channel_population_dist}, variance: {variance}, spreads: {spreads}, sparcity: {sparcity}')

        return channel_population_dist + variance + spreads + sparcity

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
