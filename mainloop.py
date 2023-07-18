import torch
import torchvision as tv
from torchmetrics.classification import MulticlassConfusionMatrix
from torch import nn
from gtaloader import TrafficDataset, GTAVTrainingFileList, GTAVValFileList, CityscapesValFileList, CityscapesTrainFileList, CityscapesTrainExtraFileList, UPCropper
import argparse
import sys
from acc_conv import AccConv2d
from dcanetv2 import DCANetV2
from BigConv import BigConvBlock
import random
import math
from collections import OrderedDict
from lsrmodels import DeeplabNW
import FeatureModifier as fn
#from FourierNormalization2d import FourierNormalization2d
#import matplotlib.pyplot as plt

LEARNING_RATE = 0.00025
BATCHES_TO_SAVE = 50
EPOCH_COUNT = 20
EPOCH_LENGTH = 1000

class ZeroLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (x * 0.0).sum() * 0.0

# class SmoothMaxLoss(nn.Module):
#     def __init__(self, alpha=1/128):
#         super().__init__()
#         self.alpha = alpha
#
#     def forward(self, x):
#         _, _, height, width = x.shape
#         distributions = x.softmax(dim=1)
#         return (1 - (distributions / self.alpha).logsumexp(dim=1) * self.alpha).mean()
#
# class NeighborLoss(nn.Module):
#     def __init__(self, channels, device, neighbor_weight=0.5):
#         super().__init__()
#         self.channels = channels
#         self.current_weight = 1 - neighbor_weight
#         self.kernel = torch.ones((channels, 1, 3, 3), device=device)
#         self.kernel[:, :, :, :] = neighbor_weight / 8.0
#         self.kernel[:, :, 1, 1] = 0

    def forward(self, x):
        distributions = x.softmax(dim=1)
        maxes, maxinds = distributions.max(dim=1, keepdim=True)
        max_indicators = torch.zeros_like(distributions).scatter(1, maxinds, 1.0)
        multiplier = -1.0/2.0
        mixed = torch.nn.functional.conv2d(max_indicators * maxes, self.kernel, padding=1, groups=self.channels)
        mixed += distributions * self.current_weight
        squares = multiplier * mixed.square()
        return squares.mean()

class WeighedMaxLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        distributions = x.softmax(dim=1)
        _, indices = distributions.max(dim=1, keepdim=True)
        class_means = 1 - distributions.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True).mean(dim=0, keepdim=True).detach().clone()
        balance_factor = class_means.mean()
        return torch.gather(-distributions * class_means, 1, indices).mean() / balance_factor

class LogLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-12

    def forward(self, x):
        distributions = x.softmax(dim=1) + self.eps
        return -distributions.log().mean()

class ExpMaxLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        distributions = x.softmax(dim=1)
        return -(distributions.max(dim=1).values.exp()).mean()

class LogMaxLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        distributions = x.softmax(dim=1)
        return -(distributions.max(dim=1).values.log()).mean()

class SquaresMaxLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxloss = MaxLoss()
        self.sqloss = SquaresLoss()

    def forward(self, x):
        return (self.maxloss(x) + self.sqloss(x)) / 2

class BatchAdaptiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-12

    def forward(self, x):
        distributions = x.softmax(dim=1)
        class_avgs = distributions.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True).mean(dim=0, keepdim=True)
        class_avgs = coerce_to_unit(class_avgs)

        return -(0.5 * distributions.square() * class_avgs + (distributions + self.eps).log() * (1 - class_avgs)).mean()

class MeanDistLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-12

    def forward(self, x):
        _, channels, _, _ = x.shape
        distributions = x.softmax(dim=1)
        return -torch.nn.functional.relu(distributions - (1.0/channels)).mean()

class NormalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        _, channels, _, _ = x.shape
        distributions = x.softmax(dim=1)
        return ((distributions / distributions.max(dim=1, keepdim=True).values).mean(dim=1) - 1).mean()

class MaxLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        distributions = x.softmax(dim=1)
        return (1 - distributions.max(dim=1).values).mean()

class Top2DiffLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        distributions = x.softmax(dim=1)
        top2 = distributions.topk(2, dim=1, sorted=False).values
        multiplier = -0.5
        return multiplier * (top2[:, 0, :, :] - top2[:, 1, :, :]).abs().mean()

class Top2SquareDiffLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        distributions = x.softmax(dim=1)
        top2 = distributions.topk(2, dim=1, sorted=False).values
        multiplier = -0.5
        return multiplier * (top2[:, 0, :, :] - top2[:, 1, :, :]).square().mean()

class SquaresLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        _, _, height, width = x.shape
        multiplier = -0.5
        distributions = x.softmax(dim=1)
        squares = multiplier * distributions.square().sum(dim=1)
        return squares.mean()

class EntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-12

    def forward(self, x):
        _, classes, _, _ = x.shape
        log_classes = math.log(classes * 1.0)
        multiplier = -1.0 / log_classes
        distributions = x.softmax(dim=1)
        entropies = 1 + multiplier * (distributions * (distributions + self.eps).log()).sum(dim=1)
        return entropies.mean()

# class InvertibleConvolution(nn.Module):
#     def __init__(self, channels, device, kernel_size=(2, 2), padding=1):
#         super().__init__()
#         self.upper_left = nn.Conv2d(channels, channels, kernel_size=kernel_size, groups=channels, padding=1)
#         self.lower_right = nn.Conv2d(channels, channels, kernel_size=kernel_size, groups=channels, padding=1)
#         self.translate_kernel = torch.tensor([[0, 0], [0, 1]], device=device)
#
#     def forward(self, x):
#         return x

class StaticDropout2d(nn.Module):
    def __init__(self, channels, p=0.5):
        super().__init__()
        self.channels = channels
        self.p = p

    def forward(self, x):
        if self.training:
            mask = torch.bernoulli(torch.ones(1, self.channels, 1, 1, device=x.device) * (1 - self.p))
            return x * mask
        return x

# # Based on torchvision/models/segmentation/_utils.py
# class CovBalancerWrapper(nn.Module):
#     def __init__(self, classifier, device):
#         super().__init__()
#         self.backbone = classifier.backbone
#         self.classifier = classifier.classifier
#         self.cov_layer = CovBalancer2d(2048, device, adapt=False)
#
#     def forward(self, x):
#         input_size = x.shape[-2:]
#         x = self.backbone(x)['out']
#         x = self.cov_layer(x)
#         x = self.classifier(x)
#         x = torch.nn.functional.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
#         return { 'out': x }

# class CovBalancer2d(nn.Module):
#     def __init__(self, channels, device, convs=5, adapt=False):
#         super().__init__()
#         self.channels = channels
#         self.iterations = nn.parameter.Parameter(torch.zeros(1).to(device), requires_grad=False)
#         self.acc_mean = nn.parameter.Parameter(torch.zeros(1).to(torch.float32).to(device), requires_grad=False)
#         self.acc_cov_matrix = nn.parameter.Parameter(torch.zeros(channels, channels).to(torch.float32).to(device), requires_grad=False)
#         self.convs = nn.ModuleList()
#         for _ in range(convs):
#             conv = nn.Conv2d(channels, channels, kernel_size=(5, 5), groups=channels)
#             conv.requires_grad = False
#             self.convs.append(conv)
#         self.avg_activations = nn.parameter.Parameter(torch.zeros(convs, 2048).to(device), requires_grad=False)
#         self.activation_vars = nn.parameter.Parameter(torch.zeros(convs, 2048).to(device), requires_grad=False)
#         self.adapt = adapt
#         self.acc_fourier = None
#         self.mse = nn.MSELoss()
#         self.eps = 1e-12
#         self.loss_mode = False
#
#     def get_cov(self):
#         return self.acc_cov_matrix / self.iterations
#
#     def get_fourier(self):
#         return self.acc_fourier / self.iterations
#
#     def get_avg_activation(self):
#         return self.avg_activations / self.iterations
#
#     def get_activation_vars(self):
#         return self.activation_vars / self.iterations
#
#     def get_only_cov(self):
#         cov = self.get_cov()
#         return cov - cov.diag().diag()
#
#     def get_corr(self):
#         cov = self.get_cov()
#         variances = cov.diag().sqrt().unsqueeze(0)
#         cross_variances = variances * variances.T
#         return cov / cross_variances
#
#     def get_mean(self):
#         return self.acc_mean / self.iterations
#
#     def predicted(self, x):
#         corr = self.get_only_cov()
#         chan_means = x.flatten(start_dim=2).mean(dim=2, keepdim=True)
#         return torch.matmul(corr, chan_means)
#
#     def activations(self, x):
#         activation_maps = []
#         for c in self.convs:
#             act = (c(x).mean(dim=3).mean(dim=2) / (x.mean(dim=3).mean(dim=2) + self.eps)).unsqueeze(1)
#             activation_maps.append(act)
#         return torch.cat(activation_maps, 1)
#
#     def loss(self, x):
#         activation_maps = self.activations(x)
#         avgs = activation_maps.mean(0)
#         vars = activation_maps.var(0)
#         avgs_diff = (avgs - self.get_avg_activation()).square().mean()
#
#         return avgs_diff
#
#     def forward(self, x):
#         if self.training and not self.loss_mode:
#             with torch.no_grad():
#                 self.iterations.copy_(self.iterations + 1)
#                 self.acc_mean.copy_(self.acc_mean + x.mean())
#                 self.acc_cov_matrix.copy_(self.acc_cov_matrix + x.flatten(start_dim=2).mean(dim=2).T.cov())
#                 activation_maps = self.activations(x)
#                 self.avg_activations += activation_maps.mean(0)
#                 self.activation_vars += activation_maps.var(0)
#
#         #if self.acc_fourier == None:
#         #    self.acc_fourier = torch.fft.rfft2(x).mean(dim=1).mean(dim=0)
#         #else:
#         #    self.acc_fourier += torch.fft.rfft2(x).mean(dim=1).mean(dim=0)
#
#         if self.adapt:
#             #pred = self.predicted(x).softmax(dim=1).unsqueeze(2)
#             #pred = torch.relu(self.predicted(x))
#             #pred = (pred / (torch.max(pred) + self.eps)).unsqueeze(2)
#             pred = self.predicted(x).unsqueeze(2)
#             batch_mean = x.mean()
#             deficit = self.get_mean() - batch_mean
#             print(f'batch_mean: {batch_mean}')
#             print(f'mean: {self.get_mean()}')
#             print(f'(self.get_mean() / batch_mean): {(self.get_mean() / batch_mean)}')
#             ratio = deficit / pred.mean()
#             appendix = pred * x * ratio
#
#             print(f'activations: {self.avg_acts(x)}')
#             print(f'activation vars: {self.act_vars(x)}')
#             output = x
#             return output
#         return x

class TeacherLoss(nn.Module):
    def __init__(self, teacher):
        super().__init__()
        teacher.eval()
        self.teacher = teacher
        self.mse = nn.MSELoss()

    def forward(self, x):
        grads = self.teacher(x).detach()
        diff = nn.functional.relu(x - grads).detach().clone()
        #diff = self.teacher(x).detach()
        #return grads.square().mean()
        return self.mse(x, diff) / 2.0

class SignLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, labels):
        ones = torch.ones_like(labels)
        maximum = labels.abs().max(dim=3, keepdim=True).values.max(dim=2, keepdim=True).values
        maximum = torch.where(maximum == 0.0, ones, maximum)
        norm_preds = preds / maximum
        norm_labels = labels / maximum
        diff = norm_preds - norm_labels
        sign_multi = nn.functional.relu(-(norm_preds * norm_labels)).sqrt()
        penalty = sign_multi * torch.abs(diff).sqrt()
        losses = penalty + diff.square()

        return losses.mean()

class KinkedLoss(nn.Module):
    def __init__(self, slope_penalty=math.e):
        super().__init__()
        self.slope_penalty = slope_penalty

    def forward(self, x, y):
        signs = torch.tanh(y * self.slope_penalty)
        preds = x * signs
        targets = y * signs
        diff = targets - preds
        inner_max = torch.maximum(diff, (self.slope_penalty * diff) + (targets - (self.slope_penalty * targets)))
        outer_max = torch.maximum(-self.slope_penalty * diff, inner_max)
        return (outer_max - targets).mean()

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
        #return source.mean() * 0.0

def coerce_to_unit(distribution):
    maxx = distribution.max(dim=1, keepdim=True).values
    minn = distribution.min(dim=1, keepdim=True).values

    return (distribution - minn) / (maxx - minn)

def mix_samples(images, predictions):
    distributions = predictions.softmax(dim=1)
    _, maxinds = distributions.max(dim=1, keepdim=True)
    #print(maxinds[0, :, :10, :10])
    #print(maxinds[1, :, :10, :10])
    max_indicators = torch.zeros_like(predictions).scatter(1, maxinds, 1.0)
    class_means = 1 - max_indicators.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True).mean(dim=0, keepdim=True).detach().clone()
    #print(max_indicators[0, :, :10, :10])
    #print(class_means)
    permutation = [*range(1, distributions.shape[0]), 0]
    label_indices = max_indicators + max_indicators[permutation, :, :, :]
    _, label_indices = (label_indices * class_means).max(dim=1)
    masks = (max_indicators * class_means).sum(dim=1, keepdim=True) > (max_indicators[permutation, :, :, :] * class_means).sum(dim=1, keepdim=True)
    #images = (images + images[permutation, :, :, :]) / 2
    images = images * masks + images[permutation, :, :, :] * masks.logical_not()

    #print(label_indices[0, :10, :10])
    # torch.zeros_like(a).scatter(1, maxinds, 1.0)
    # rotate = a[:,[-1, *range(0, dims - 1)] , :, :,]
    return (images, label_indices)

def pixel_accuracy(confusion_matrix):
    computed = confusion_matrix.compute()
    return float(computed.diagonal().sum()) / float(computed.sum())

def IoU(confusion_matrix, ignore_index=255):
    computed = confusion_matrix.compute()
    ious = computed.diagonal().clone().to(torch.float)
    for i in range(ious.numel()):
        if i == ignore_index:
            ious[i] = 0.0
        else:
            ious[i] = float(ious[i]) / float(computed[i,:].sum() + computed[:,i].sum() - ious[i])
    return ious

def mIoU(ious, ignore_index=255):
    if ignore_index == None:
        return float(ious.sum()) / float(ious.numel())

    if ignore_index in range(ious.numel()):
        ious[ignore_index] = 0.0
    return float(ious.sum()) / float(ious.numel() - 1)

def plot_images_labels(batch_images, batch_labels):
    f, axarr = plt.subplots(2, batch_images.shape[0])
    for i in range(batch_images.shape[0]):
        axarr[0, i].imshow(batch_images[i].permute(1, 2, 0))
        axarr[1, i].imshow(batch_labels[i])

    plt.show()

def to_one_hot(indices, device, classes=19, ignore_index=255):
    #print(indices.shape)
    batch_size, height, width = indices.shape
    indices[indices == ignore_index] = classes
    one_hot = torch.zeros(batch_size, classes+1, height, width, device=device).to(torch.float32)
    #print(one_hot.shape)

    return one_hot.scatter(1, indices.unsqueeze(1), 1.0)[:, :-1, :, :]

def train_epoch(dataloader, optimizer, classifier, loss_fun, device, scheduler, unsupervised=False, lock_backbone=False):
    batch_count = 0
    classifier.train()
    if lock_backbone:
        classifier.backbone.eval()
    total_loss = 0
    for ix, (batch_images, batch_labels) in enumerate(dataloader):
        #plot_images_labels(batch_images, batch_labels)
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

def train_reverse(dataloader, optimizer, classifier, reverser, loss_fun, device, scheduler, ignore_index=255):
    batch_count = 0
    classifier.eval()
    reverser.train()
    dataloader.train_augmentations = False
    softmax = nn.Softmax(dim=1)
    total_loss = 0
    for ix, (batch_images, _) in enumerate(dataloader):
        #plot_images_labels(batch_images, batch_labels)
        optimizer.zero_grad()
        batch_images = batch_images.to(device)
        pred_maps = None
        with torch.no_grad():
            pred_maps = softmax(classifier(batch_images)['out'])
        predictions = reverser(pred_maps)

        loss = loss_fun(predictions, batch_images)

        loss.backward()
        optimizer.step()
        if batch_count % 10 == 0:
            print(f'Reverse training batch: {batch_count}, Loss: {loss.item()}, learning rate: {scheduler.get_last_lr()}')
        total_loss += loss.item()
        batch_count += 1
        scheduler.step()
        if EPOCH_LENGTH and batch_count >= EPOCH_LENGTH:
            break
    total_loss /= float(batch_count)
    print(f'Mean reverse training loss for epoch: {total_loss}')

def train_augment_reverse(dataloader, optimizer, classifier, reverser, loss_fun, device, scheduler, aug_weight=0.1, noise_factor=0.2, ignore_index=255):
    batch_count = 0
    classifier.train()
    reverser.eval()
    softmax = nn.Softmax(dim=1)
    dataloader.train_augmentations = False
    total_loss = 0
    for ix, (batch_images, batch_labels) in enumerate(dataloader):
        #plot_images_labels(batch_images, batch_labels)
        optimizer.zero_grad()
        batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
        reversed = None
        with torch.no_grad():
            pred_maps = softmax(classifier(batch_images)['out'])
            reversed = reverser(pred_maps)
        classifier.zero_grad()
        predictions = classifier(aug_weight * reversed + (1 - aug_weight) * batch_images)['out']
        loss = loss_fun(predictions, batch_labels)

        loss.backward()
        optimizer.step()
        if batch_count % 10 == 0:
            print(f'Training batch (reverse aug): {batch_count}, Loss: {loss.item()}, learning rate: {scheduler.get_last_lr()}')
        total_loss += loss.item()
        batch_count += 1
        scheduler.step()
        if EPOCH_LENGTH and batch_count >= EPOCH_LENGTH:
            break
    total_loss /= float(batch_count)
    print(f'Mean training loss for epoch: {total_loss}')

def train_unsupervised_end_to_end(dataloader, us_dataloader, optimizer, classifier, loss_fun, unsuperv_loss_fun, device, scheduler, lam=0.001, self_train=False):
    print(f'Training end to end with unsupervised loss {unsuperv_loss_fun.__class__.__name__}, lambda={lam}')
    self_train_lam = 0.1
    if self_train:
        print(f'Using self training with self_train_lambda={self_train_lam}')
    batch_count = 0
    classifier.train()
    total_loss = 0
    dl_len = len(dataloader)
    udl_len = len(us_dataloader)
    dl_enum = enumerate(dataloader)
    udl_enum = enumerate(us_dataloader)
    for i in range(min(dl_len, udl_len, EPOCH_LENGTH)):
        optimizer.zero_grad()
        ix, (batch_images, batch_labels) = next(dl_enum)
            #plot_images_labels(batch_images, batch_labels)
        batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
        predictions = classifier(batch_images)['out']

        superv_loss = loss_fun(predictions, batch_labels)

        ix, (batch_images, _) = next(udl_enum)
        batch_images = batch_images.to(device)
        predictions = classifier(batch_images)['out']

        loss = superv_loss + lam * unsuperv_loss_fun(predictions)

        if self_train:
            batch_images, batch_labels = mix_samples(batch_images, predictions)
            predictons = classifier(batch_images)['out']
            loss += self_train_lam * loss_fun(predictions, batch_labels)
        loss.backward()
        optimizer.step()
        if batch_count % 10 == 0:
            print(f'Training unsupervised e2e batch (unsuperv loss): {batch_count}, Loss: {loss.item()}, learning rate: {scheduler.get_last_lr()}')
                #plt.imshow(predictions[0][0].detach().numpy())
                #plt.show()
        total_loss += loss.item()
        batch_count += 1
        scheduler.step()
        if EPOCH_LENGTH and batch_count >= EPOCH_LENGTH:
            break
    total_loss /= float(batch_count)
    print(f'Mean training loss for epoch (unsupervised e2e): {total_loss}')

def train_teacher(dataloader, optimizer, backbone, dumb_backbone, head, teacher, head_loss_fun, teacher_loss_fun, device, scheduler):
    print('Starting teacher training.')
    batch_count = 0
    teacher.train()
    head.train()
    backbone.eval()
    dumb_backbone.eval()

    total_loss = 0
    for ix, (batch_images, batch_labels) in enumerate(dataloader):
        optimizer.zero_grad()
        batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
        features = None
        if random.random() < 0.5:
            features = backbone(batch_images)['out'].detach().clone()
        else:
            features = dumb_backbone(batch_images)['out'].detach().clone()
        features.requires_grad = True
        head_loss_fun(head(features, batch_images.shape[-2:])['out'], batch_labels).backward()

        target_grad = features.grad.detach().clone()
        preds = teacher(features)
        target_grad_magnitude = target_grad.abs().mean()
        target_grad = target_grad / target_grad_magnitude

        if random.randrange(200) == 0:
            print(f'pred: {preds.flatten(start_dim=1).mean(dim=1)}, true: {target_grad.flatten(start_dim=1).mean(dim=1)}')

        teacher_loss = teacher_loss_fun(preds, target_grad)
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
    ps = []

    total_loss = 0
    for ix, (batch_images, _) in enumerate(dataloader):
        optimizer.zero_grad()
        batch_images = batch_images.to(device)
        features = backbone(batch_images)['out']

        loss = loss_fun(features)
        loss.backward()

        #if len(ps) > 0:
        #    i = 0
        #    for p in backbone.parameters():
        #        print(ps[i] - p)
        #        i += 1
        #    ps = []
        #for p in backbone.parameters():
        #    ps.append(p.detach())

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

def fourier_stats_run(dataloader, fourier_layer, device):
    fourier_layer.train()
    batch_count = 0
    for batch_images, _ in dataloader:
        batch_images = batch_images.to(device)
        fourier_layer(batch_images)
        if batch_count % 10 == 0:
            print(f'Fourier stats batch {batch_count}')
        batch_count += 1
    print('Fourier stats ran for epoch')

def validate(dataloader, classifier, num_classes, device):
    classifier.eval()
    print('Validating...')
    batch_count = 0
    total_batches = len(dataloader)
    confusion_matrix = MulticlassConfusionMatrix(num_classes, ignore_index=255).to(device)
    with torch.no_grad():
        for batch_images, batch_labels in dataloader:
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            predictions = classifier(batch_images)['out']
            normalized_masks = predictions.softmax(dim=1)
            confusion_matrix.update(normalized_masks, batch_labels)

            if batch_count % 10 == 0:
                print(f'Validation batch: {batch_count}, Pixel accuracy so far: {pixel_accuracy(confusion_matrix)}')
            batch_count += 1
    ious = IoU(confusion_matrix, ignore_index=255)
    miou = mIoU(ious, ignore_index=255)
    print(f'Pixel accuracy: {pixel_accuracy(confusion_matrix)}')
    print(f'mIoU: {miou}')
    print(f'Class IoUs: {ious}')

def load_gtav_set(dataset_dir, device='cpu'):
    filelist = GTAVTrainingFileList(dataset_dir, training_split_ratio=0.95)
    val_filelist = GTAVValFileList(dataset_dir, training_split_ratio=0.97)
    assert len(set(filelist) & set(val_filelist)) == 0
    # orig size (704, 1264)?
    up_cropper = UPCropper(device=device, crop_size=(768, 768), samples=1)

    dataset = TrafficDataset(filelist, resize=(1052, 1914), train_augmentations=True, cropper=up_cropper, device=device)
    val_dataset = TrafficDataset(val_filelist, resize=(1052, 1914), device=device)

    return dataset, val_dataset

def load_cityscapes_set(dataset_dir, device='cpu'):
    filelist = CityscapesTrainFileList(dataset_dir)
    val_filelist = CityscapesValFileList(dataset_dir)
    #print(len(filelist))
    #for f in filelist:
    #    print(f[0])
    assert len(set(filelist) & set(val_filelist)) == 0

    up_cropper = UPCropper(device=device, crop_size=(768, 768), samples=1)

    size = (1024, 2048)
    print(f'Using cityscapes size {size}')

    dataset = TrafficDataset(filelist, resize=size, train_augmentations=True, cropper=up_cropper, device=device)
    val_dataset = TrafficDataset(val_filelist, resize=size, device=device)
    #val_dataset = TrafficDataset(val_filelist, resize=(512, 1024), crop_size=(512, 1024))

    return dataset, val_dataset

def load_cityscapes_unsupervised_set(dataset_dir, device='cpu'):
    filelist = CityscapesTrainExtraFileList(dataset_dir)
    val_filelist = CityscapesValFileList(dataset_dir)
    #print(len(filelist))
    #for f in filelist:
    #    print(f[0])
    assert len(set(filelist) & set(val_filelist)) == 0

    up_cropper = UPCropper(device=device, crop_size=(768, 768), samples=1)

    size = (1024, 2048)
    print(f'Using cityscapes size {size}')

    dataset = TrafficDataset(filelist, resize=size, train_augmentations=True, cropper=up_cropper, device=device, allow_missing_labels=True)
    val_dataset = TrafficDataset(val_filelist, resize=size, device=device)
    #val_dataset = TrafficDataset(val_filelist, resize=(512, 1024), crop_size=(512, 1024))

    return dataset, val_dataset

def start(save_file_name=None, load_file_name=None, load_backbone=None, load_model=None, dataset_type='gtav', dataset_dir='../datasetit/gtav/', adaptation_dir='../datasetit/cityscapes/', device='cpu', only_adapt=False, unsupervised=False, lock_backbone=False, teacher_mode=False, student_mode=False, load_teacher=None, model_type='rn50', batch_size=8, negative_model=False, cov_layer_adapt=False, cov_layer_loss=False, init_fourier=False, fourier_stats=False, add_fourier=False, load_init_fourier=None, us_e2e_loss=None, us_dataset_dir=None, us_dataset_type=None, us_lambda=0.001, us_self_train=False, reverse_mode_train=False, load_reverser=None, train_with_reverser=False):

    batch_size = int(batch_size)
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
        raise Exception(f'Unknown dataset type {dataset_type}')

    us_dataset, us_val_dataset = (None, None)
    us_dataloader, us_validation_dataloader = (None, None)
    if us_dataset_dir and us_dataset_type:
        if us_dataset_type == 'gtav':
            us_dataset, us_val_dataset = load_gtav_set(us_dataset_dir, device=device)
        #loss_weights = torch.Tensor([0.0, 2.7346e+00, 1.0929e+01, 5.3414e+00, 4.7041e+01, 1.3804e+02,
        #                             8.5594e+01, 6.6778e+02, 1.0664e+03, 1.1754e+01, 4.1259e+01, 6.5842e+00,
        #                             2.4710e+02, 3.0451e+03, 3.4462e+01, 7.5414e+01, 2.4990e+02, 1.4041e+03,
        #                             2.7946e+03, 1.7960e+04]).to(device)
            print(f'Loaded GTAV dataset at {us_dataset_dir} for unsupervised')
        elif us_dataset_type == 'cityscapes':
            us_dataset, us_val_dataset = load_cityscapes_unsupervised_set(us_dataset_dir)
            if model_type == 'rn50':
                print('Loading cityscapes supervised instead for smaller model')
                us_dataset, us_val_dataset = load_cityscapes_set(us_dataset_dir)
            print(f'Loaded Cityscapes dataset at {us_dataset_dir} for unsupervised')
        else:
            raise Exception(f'Unknown dataset type {us_dataset_type} for unsupervised')

        us_dataloader = torch.utils.data.DataLoader(us_dataset, drop_last=True, batch_size=batch_size, shuffle=True)
        us_validation_dataloader = torch.utils.data.DataLoader(us_val_dataset, drop_last=True, batch_size=batch_size)

        print(f'Dataloaders initialized for unsupervised (type {us_dataset_type})')

    num_classes = dataset.COLOR_COUNT
    dataloader = torch.utils.data.DataLoader(dataset, drop_last=True, batch_size=batch_size, shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(val_dataset, drop_last=True, batch_size=batch_size)
    print('Dataloader initialized')
    # params 11029075 (mobilenetv3)
    # params 10413651 (DCANet)

    classifier = None
    if model_type == 'rn101':
        print('Loaded ResNet101 based model')
        classifier = tv.models.segmentation.deeplabv3_resnet101(num_classes = num_classes)
    elif model_type == 'rn101os16':
        print('Loaded ResNet101 based model')
        classifier = tv.models.segmentation.deeplabv3_resnet101(num_classes = num_classes)
        asps = list(list(list(classifier.classifier.children())[0].children())[0].children())

        asp1 = list(asps[1].children())[0]
        asp1.padding = (6, 6)
        asp1.dilation = (6, 6)

        asp2 = list(asps[2].children())[0]
        asp2.padding = (12, 12)
        asp2.dilation = (12, 12)

        asp3 = list(asps[3].children())[0]
        asp3.padding = (18, 18)
        asp3.dilation = (18, 18)

        print(classifier)
    elif model_type == 'lsr':
        print('LSR model loaded with RN101 backbone')
        classifier = DeeplabNW(num_classes = num_classes, backbone='ResNet101', pretrained=False)
    elif model_type == 'cov':
        print('Loaded ResNet50 based model with cov layer')
        classifier = tv.models.segmentation.deeplabv3_resnet50(num_classes = num_classes)
        classifier = CovBalancerWrapper(classifier, device)
    else:
        print('Loaded ResNet50 based model')
        classifier = tv.models.segmentation.deeplabv3_resnet50(num_classes = num_classes)
    # classifier = DeeplabNW(num_classes = dataset.COLOR_COUNT, backbone='resnet50', pretrained=False)

    if add_fourier:
        print('Adding Fourier layer')
        classifier.backbone = nn.Sequential(FourierNormalization2d(3, 256, 256).to(device), classifier.backbone)

    teacher = None
    untrained_backbone = None

    if negative_model:
        if model_type == 'rn101':
            print('Loaded ResNet101 based backbone for negative model base')
            untrained_backbone = tv.models.segmentation.deeplabv3_resnet101(num_classes = 2048).backbone
        else:
            print('Loaded ResNet50 based backbone for negative model base')
            untrained_backbone = tv.models.segmentation.deeplabv3_resnet50(num_classes = 2048).backbone

        untrained_backbone = untrained_backbone.to(device)

    reverser = None

    if reverse_mode_train or train_with_reverser:
        print('Initializing reverse mode network')
        reverser = fn.FeatureModifier(in_channels=19, out_channels=3, sum_initial_layer=False).to(device)
        dataset.train_augmentations = False
        us_dataset.train_augmentations = False

        if load_reverser:
            print(f'Loading reverser from {load_reverser}')
            reverser.load_state_dict(torch.load(load_reverser, map_location=device)['reverser_state_dict'])

    if teacher_mode:
        print('Initializing teacher network')
        untrained_backbone = None
        if model_type == 'rn101':
            print('Loaded ResNet101 based backbone for teacher training')
            untrained_backbone = tv.models.segmentation.deeplabv3_resnet101(num_classes = 2048).backbone
        else:
            print('Loaded ResNet50 based backbone for teacher training')
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
        teacher.load_state_dict(torch.load(load_teacher, map_location=torch.device('cpu'))['teacher_state_dict'])

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
    unsuperv_loss_fun = None
    optim_params = []

    if unsupervised:
        loss_fun = USSegLoss()
        optim_params = [{'params': classifier.parameters(), 'lr': LEARNING_RATE }]
    elif lock_backbone:
        loss_fun = nn.CrossEntropyLoss(ignore_index=255, weight=loss_weights)
        optim_params = [{'params': classifier.backbone.parameters(), 'lr': 0 },
                        { 'params': classifier.classifier.parameters(), 'lr': 10 * LEARNING_RATE }]
    elif teacher_mode:
        loss_fun = nn.CrossEntropyLoss(ignore_index=255, weight=loss_weights)
        optim_params = [{'params': teacher.parameters(), 'lr': LEARNING_RATE}]
    elif student_mode:
        loss_fun = TeacherLoss(teacher)
        optim_params = [{ 'params': classifier.backbone.parameters(), 'lr': LEARNING_RATE }]
    elif model_type == 'lsr':
        loss_fun = nn.CrossEntropyLoss(ignore_index=255, weight=loss_weights)
        optim_params = [{'params': classifier.parameters(), 'lr': LEARNING_RATE * 5 }]
    elif cov_layer_loss:
        loss_fun = classifier.cov_layer.loss
        optim_params = [{'params': classifier.backbone.parameters(), 'lr': LEARNING_RATE},
                        {'params': classifier.cov_layer.parameters(), 'lr': 0}]
    elif model_type == 'cov':
        loss_fun = nn.CrossEntropyLoss(ignore_index=255, weight=loss_weights)
        optim_params = [{'params': classifier.parameters(), 'lr': 0 }]
    elif reverse_mode_train:
        loss_fun = nn.MSELoss()
        optim_params = [{'params': reverser.parameters(), 'lr': LEARNING_RATE }]
    elif train_with_reverser:
        loss_fun = nn.CrossEntropyLoss(ignore_index=255, weight=loss_weights)
        optim_params = [{'params': classifier.backbone.parameters(), 'lr': LEARNING_RATE },
                        { 'params': classifier.classifier.parameters(), 'lr': 10 * LEARNING_RATE }]
    elif us_e2e_loss:
        loss_fun = nn.CrossEntropyLoss(ignore_index=255, weight=loss_weights)
        optim_params = [{'params': classifier.backbone.parameters(), 'lr': LEARNING_RATE },
                        { 'params': classifier.classifier.parameters(), 'lr': 10 * LEARNING_RATE }]
        if us_e2e_loss == 'entropy':
            unsuperv_loss_fun = EntropyLoss()
        elif us_e2e_loss == 'squares':
            unsuperv_loss_fun = SquaresLoss()
        elif us_e2e_loss == 'meandist':
            unsuperv_loss_fun = MeanDistLoss()
        elif us_e2e_loss == 'max':
            unsuperv_loss_fun = MaxLoss()
        elif us_e2e_loss == 'squaresmax':
            unsuperv_loss_fun = SquaresMaxLoss()
        elif us_e2e_loss == 'smoothmax':
            unsuperv_loss_fun = SmoothMaxLoss()
        elif us_e2e_loss == 'neighbor':
            unsuperv_loss_fun = NeighborLoss(num_classes, device)
        elif us_e2e_loss == 'weightedmax':
            unsuperv_loss_fun = WeighedMaxLoss()
        elif us_e2e_loss == 'batchadaptive':
            unsuperv_loss_fun = BatchAdaptiveLoss()
        elif us_e2e_loss == 'logmax':
            unsuperv_loss_fun = LogMaxLoss()
        elif us_e2e_loss == 'expmax':
            unsuperv_loss_fun = ExpMaxLoss()
        elif us_e2e_loss == 'top2diff':
            unsuperv_loss_fun = Top2DiffLoss()
        elif us_e2e_loss == 'top2squarediff':
            unsuperv_loss_fun = Top2SquareDiffLoss()
        elif us_e2e_loss == 'normal':
            unsuperv_loss_fun = NormalLoss()
        elif us_e2e_loss == 'log':
            unsuperv_loss_fun = LogLoss()
        elif us_e2e_loss == 'zero':
            unsuperv_loss_fun = ZeroLoss()
        else:
            raise ValueError(f'Unrecognized unsupervised loss "{us_e2e_loss}"')
    else:
        loss_fun = nn.CrossEntropyLoss(ignore_index=255, weight=loss_weights)
        optim_params = [{'params': classifier.backbone.parameters(), 'lr': LEARNING_RATE },
                        { 'params': classifier.classifier.parameters(), 'lr': 10 * LEARNING_RATE }]

    # 10x LR for classifier, 1x for backbone
    optimizer = torch.optim.SGD(optim_params, momentum=0.9, weight_decay=0.0005)
    total_iters = EPOCH_COUNT * EPOCH_LENGTH
    print(f'Running for total number of iterations of {total_iters}')
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_iters, power=0.9)
    epoch = 0

    if load_file_name:
        print(f'Loading checkpoint from file {load_file_name}...')
        checkpoint = torch.load(load_file_name)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        if teacher_mode and 'teacher_state_dict' in checkpoint:
            teacher.load_state_dict(checkpoint['teacher_state_dict'])
        print('Done loading')

    if teacher_mode and load_teacher:
        print(f'Loading teacher from {load_teacher}')
        teacher.load_state_dict(torch.load(load_teacher)['teacher_state_dict'])

    if load_model:
        print(f'Loading model weights separately from {load_model}')
        if model_type == 'cov':
            classifier.load_state_dict(torch.load(load_model)['model_state_dict'], strict=False)
        elif unsupervised:
            print(f'Loading model weights in unsupervised mode')
            s_dict = []
            for k, v in torch.load(load_model)['model_state_dict'].items():
                if k.startswith('backbone.'):
                    s_dict.append((k.split('.', 1)[1], v))
            classifier.load_state_dict(OrderedDict(s_dict))
        else:
            classifier.load_state_dict(torch.load(load_model, map_location=device)['model_state_dict'])

    if load_backbone and model_type == 'lsr':
        print(f'Loading backbone weights from pure weights file {load_backbone}')
        s_dict = torch.load(load_backbone)['state_dict']
        s_dict = OrderedDict([(k.split('.', 1)[1], v) if k.startswith('module.') else (k, v) for k, v in s_dict.items()])
        classifier.load_state_dict(s_dict, strict=False)
    elif load_backbone:
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

    if cov_layer_adapt:
        print('Switching Cov layer to adapt mode')
        cov_module = classifier.cov_layer
        cov_module.adapt = True
        print(f'cov_module.get_cov()[:10, :10]: {cov_module.get_cov()[:10, :10]}')
        print(f'cov_module.get_mean(): {cov_module.get_mean()}')
        print(cov_module.get_avg_activation())
        print(cov_module.get_activation_vars())
        def print_feature_stats(module, input, output):
            out = output

            #print(cov[:10,:10])
            #print(predicted[:, :10])
            #print(out.flatten(start_dim=2).mean(dim=2)[:, :10])
            #print(predicted.softmax(dim=1)[:, :10])
            #print(out.flatten(start_dim=2).mean(dim=2).softmax(dim=1)[:, :10])
            #print(cov_module.get_mean())
            #vals = output['out']
            #print(f'vals.shape: {vals.shape}')
            print(f'out.mean(): {out.mean()}')
            #print(f'predicted.mean(): {predicted.mean()}')
            #print(f'predicted.flatten(start_dim=1).mean(dim=1): {predicted.flatten(start_dim=1).mean(dim=1)}')
            #print(f'out.flatten(start_dim=1).mean(dim=1): {out.flatten(start_dim=1).mean(dim=1)}')
            #print(f'vals.flatten(start_dim=2).var(dim=2).mean(): {vals.flatten(start_dim=2).var(dim=2).mean()}')
            #print(f'vals.flatten(start_dim=2).mean(dim=2).var(dim=1).mean(): {vals.flatten(start_dim=2).mean(dim=2).var(dim=1).mean()}')

        #classifier.backbone.load_state_dict(model_params)
        cov_module.register_forward_hook(print_feature_stats)


    if negative_model:
        #default_params = untrained_backbone.state_dict()
        #model_params = classifier.backbone.state_dict()

        #for name, param in default_params.items():
        #    if not "weight" in name:
        #        continue
        #    model_param = model_params[name]
        #    model_params[name] = -(model_param - param) + param

        classifier.backbone.add_module('cov', CovBalancer2d(2048, device, adapt=True))

        print('(actually normal model)')
        def print_feature_stats(module, input, output):
            out = output['out']
            cov_module = module.get_submodule('cov')
            cov = cov_module.get_cov()
            predicted = cov_module.predicted(out).flatten(start_dim=1)
            print(cov[:10,:10])
            print(predicted[:, :10])
            print(out.flatten(start_dim=2).mean(dim=2)[:, :10])
            print(predicted.softmax(dim=1)[:, :10])
            print(out.flatten(start_dim=2).mean(dim=2).softmax(dim=1)[:, :10])
            print(cov_module.get_mean())
            #vals = output['out']
            #print(f'vals.shape: {vals.shape}')
            print(f'out.mean(): {out.mean()}')
            print(f'predicted.mean(): {predicted.mean()}')
            print(f'predicted.flatten(start_dim=1).mean(dim=1): {predicted.flatten(start_dim=1).mean(dim=1)}')
            print(f'out.flatten(start_dim=1).mean(dim=1): {out.flatten(start_dim=1).mean(dim=1)}')
            #print(f'vals.flatten(start_dim=2).var(dim=2).mean(): {vals.flatten(start_dim=2).var(dim=2).mean()}')
            #print(f'vals.flatten(start_dim=2).mean(dim=2).var(dim=1).mean(): {vals.flatten(start_dim=2).mean(dim=2).var(dim=1).mean()}')

        #classifier.backbone.load_state_dict(model_params)
        classifier.backbone.register_forward_hook(print_feature_stats)
    if init_fourier:
        print('Initializing Fourier layer')
        classifier.backbone = nn.Sequential(FourierNormalization2d(3, 20, 20).to(device), classifier.backbone)
    elif load_init_fourier:
        print(f'Initializing Fourier layer from {load_init_fourier}')
        fourier_norm = FourierNormalization2d(3, 100, 100).to(device)
        fourier_norm.load_state_dict(torch.load(load_init_fourier, map_location=device))
        fourier_norm.width = 20
        fourier_norm.height = 20
        classifier.backbone = nn.Sequential(fourier_norm, classifier.backbone)

    fourier_layer_only = None
    if fourier_stats:
        print('Getting fourier stats')
        fourier_layer_only = FourierNormalization2d(3, 100, 100).to(device)

    print('Network initialized')

    print('Starting optimization')

    if only_adapt:
        adaptation_filelist = CityscapesValFileList(adaptation_dir)
        adaptation_dataloader = torch.utils.data.DataLoader(TrafficDataset(adaptation_filelist), drop_last=True, batch_size=batch_size)
        print('== DA Validation ==')
        validate(adaptation_dataloader, classifier, num_classes, device)
        print('== DA Validation done ==')
        return
    if fourier_stats:
        while epoch < EPOCH_COUNT:
            fourier_stats_run(dataloader, fourier_layer_only, device)

            print(f'saving to {save_file_name}')
            torch.save(fourier_layer_only.state_dict(), save_file_name)
            epoch += 1
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
            train_teacher(dataloader, optimizer, backbone, untrained_backbone, head, teacher, loss_fun, KinkedLoss(), device, scheduler)
        elif student_mode:
            backbone = classifier.backbone
            for p in classifier.classifier.parameters():
                p.requires_grad = False
            student_train(dataloader, optimizer, backbone, loss_fun, device, scheduler)
        elif cov_layer_loss:
            print('Using cov layer loss')
            backbone = classifier.backbone
            classifier.cov_layer.train()
            classifier.cov_layer.loss_mode = True
            for p in classifier.classifier.parameters():
                p.requires_grad = False
            student_train(dataloader, optimizer, backbone, loss_fun, device, scheduler)
        elif us_e2e_loss:
            train_unsupervised_end_to_end(dataloader, us_dataloader, optimizer, classifier, loss_fun, unsuperv_loss_fun, device, scheduler, lam=us_lambda, self_train=us_self_train)
        elif reverse_mode_train:
            train_reverse(us_dataloader, optimizer, classifier, reverser, loss_fun, device, scheduler)
        elif train_with_reverser:
            train_augment_reverse(dataloader, optimizer, classifier, reverser, loss_fun, device, scheduler)
        elif not negative_model:
            train_epoch(dataloader, optimizer, classifier, loss_fun, device, scheduler, unsupervised, lock_backbone)
        if not unsupervised and not teacher_mode and not reverse_mode_train:
            validate(validation_dataloader, classifier, num_classes, device)

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
            elif reverse_mode_train:
                save_dict['reverser_state_dict'] = reverser.state_dict()
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
    parser.add_argument('--device', dest='device', default='cpu')
    parser.add_argument('--load-backbone', dest='load_backbone')
    parser.add_argument('--load-model', dest='load_model')
    parser.add_argument('--unsupervised', dest='unsupervised', action='store_true')
    parser.add_argument('--lock-backbone', dest='lock_backbone', action='store_true')
    parser.add_argument('--only-adapt', dest='only_adapt', action='store_true')
    parser.add_argument('--teacher-mode', dest='teacher_mode', action='store_true')
    parser.add_argument('--student-mode', dest='student_mode', action='store_true')
    parser.add_argument('--load-teacher', dest='load_teacher')
    parser.add_argument('--model-type', dest='model_type', default='rn50')
    parser.add_argument('--batch-size', dest='batch_size', default='8')
    parser.add_argument('--negative-model', dest='negative_model', action='store_true')
    parser.add_argument('--cov-layer-adapt', dest='cov_layer_adapt', action='store_true')
    parser.add_argument('--cov-layer-loss', dest='cov_layer_loss', action='store_true')
    parser.add_argument('--init-fourier', dest='init_fourier', action='store_true')
    parser.add_argument('--fourier-stats', dest='fourier_stats', action='store_true')
    parser.add_argument('--add-fourier', dest='add_fourier', action='store_true')
    parser.add_argument('--load-init-fourier', dest='load_init_fourier')
    parser.add_argument('--us-dataset', dest='us_dataset_dir')
    parser.add_argument('--us-dataset-type', dest='us_dataset_type')
    parser.add_argument('--us-e2e-loss', dest='us_e2e_loss')
    parser.add_argument('--us-lambda', dest='us_lambda', type=float, default=0.001)
    parser.add_argument('--us-self-train', dest='us_self_train', action='store_true')
    parser.add_argument('--reverse-mode-train', dest='reverse_mode_train', action='store_true')
    parser.add_argument('--load-reverser', dest='load_reverser')
    parser.add_argument('--train-with-reverser', dest='train_with_reverser', action='store_true')
    args = parser.parse_args()
    start(args.save_file_name, args.load_file_name, args.load_backbone, args.load_model, args.dataset_type, args.dataset_dir, args.adaptset_dir, args.device, args.only_adapt, args.unsupervised, args.lock_backbone, args.teacher_mode, args.student_mode, args.load_teacher, args.model_type, args.batch_size, args.negative_model, args.cov_layer_adapt, args.cov_layer_loss, args.init_fourier, args.fourier_stats, args.add_fourier, args.load_init_fourier, args.us_e2e_loss, args.us_dataset_dir, args.us_dataset_type, args.us_lambda, args.us_self_train, args.reverse_mode_train, args.load_reverser, args.train_with_reverser)
