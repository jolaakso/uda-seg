import torch
import argparse
import sys
from torch import nn
import torchvision as tv
from gtaloader import GTAVDataset

BATCH_SIZE = 5
BATCHES_TO_SAVE = 50
EPOCH_COUNT = 10

class SegmentClassifier(nn.Module):
    def __init__(self, output_classes):
        super().__init__()
        self.featurizer_head = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.CELU()
        )
        self.featurizer_compressor = nn.Sequential(
            nn.Conv2d(512, 64, 3, padding=1),
            nn.CELU()
        )

        self.expander = nn.Sequential(
            nn.ConvTranspose2d(64, output_classes * 2, 5, padding=0, stride=4, output_padding=2),
            nn.CELU(),
            nn.ConvTranspose2d(output_classes * 2, output_classes, 5, padding=0, stride=4, output_padding=3),
            nn.CELU(),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        residual = self.featurizer_head(x)
        x = self.featurizer_compressor(x + residual)
        return self.expander(x)

class Segmenter(nn.Module):
    def __init__(self, output_classes):
        super().__init__()
        weights = tv.models.SqueezeNet1_1_Weights.DEFAULT
        # (3, 704, 1264) -> (512, 44, 79)
        squeezenet_conv_layers = list(tv.models.squeezenet1_1(weights=weights).children())[0]
        self.backbone = nn.Sequential(*squeezenet_conv_layers)
        self.classifier = SegmentClassifier(output_classes)

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)

def start(save_file_name=None, load_file_name=None):
    dataset = GTAVDataset('../datasetit/gtav/')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print('Dataloader initialized')
    segmenter = Segmenter(dataset.COLOR_COUNT * 2)
    classifier = segmenter.classifier

    if load_file_name:
        print(f'Loading model from file {load_file_name}...')
        classifier.load_state_dict(torch.load(load_file_name))
        print('Done loading')

    classifier.train()
    print('Network initialized')
    loss_fun = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.3)
    print('Starting optimization')
    if save_file_name:
        print(f'Will save the model every {BATCHES_TO_SAVE} batches to {save_file_name}')
    total_count = 1
    for epoch in range(EPOCH_COUNT):
        print(f'Epoch: {epoch}')
        batch_count = 1
        for ix, batch in enumerate(dataloader):
            optimizer.zero_grad()
            batch_images, batch_labels = batch
            predictions = segmenter(batch_images)
            loss = loss_fun(predictions, batch_labels)
            loss.backward()
            optimizer.step()
            if batch_count % 3 == 0:
                print(f'Batch: {batch_count}, Loss: {loss.item()}')
            if save_file_name and total_count % BATCHES_TO_SAVE == 0:
                print(f'Saving model to file {save_file_name}...')
                torch.save(classifier.state_dict(), save_file_name)
                print('Done saving')
            batch_count += 1
            total_count += 1
        scheduler.step()

if __name__ == "__main__":
    save_file = None
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--save', dest='save_file_name')
    parser.add_argument('--load', dest='load_file_name')
    args = parser.parse_args()
    start(args.save_file_name, args.load_file_name)
