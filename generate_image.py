import torch
import torchvision as tv
import argparse
import gtaloader
import matplotlib.pyplot as plt
from DCANet import DCANet

def print_class_stats(classes):
    print(f'{classes.mean(2).mean(1)} classes.mean(2).mean(1)')
    print(f'{torch.var(classes.mean(2).mean(1))} torch.var(classes.mean(2).mean(1))')
    print(f'{classes.mean()} classes.mean()')
    print(f'{classes.flatten(start_dim=1).var(dim=1)} classes.flatten(start_dim=1).var(dim=1)')
    print(f'{classes.flatten(start_dim=1).var(dim=1).mean()} classes.flatten(start_dim=1).var(dim=1).mean()')
    print(f'{classes.mean(0).flatten(start_dim=0).var()} classes.mean(0).flatten(start_dim=0).var()')
    print(f'{classes.flatten(start_dim=1).mean(1).var(dim=0)} classes.flatten(start_dim=1).mean(1).var(dim=0)')
    print(f'{classes.mean(0).flatten(start_dim=0).var()} classes.flatten(start_dim=1).mean(1).var(dim=0)')
    print(f'{torch.var(classes.mean(2), dim=1)} torch.var(classes.mean(2), dim=1)')
    print(f'{torch.var(classes.mean(1), dim=1)} torch.var(classes.mean(1), dim=1)')


def segment_image(image, model):
    output = model(image.unsqueeze(0))['out']
    normalized = torch.nn.functional.softmax(output, dim=1)
    _, argmaxes = torch.max(normalized, dim=1, keepdim=True)
    bools = torch.zeros_like(normalized).scatter(1, argmaxes, 1.0).to(torch.bool).squeeze(0)
    # add colors
    masks = tv.utils.draw_segmentation_masks(tv.transforms.functional.convert_image_dtype(image, dtype=torch.uint8), bools, colors=gtaloader.VALID_CLASS_COLORS, alpha=1.0)
    return masks

def start(model_weights, image_file_name, output_file):
    classifier = tv.models.segmentation.deeplabv3_resnet101(num_classes = 19).to('cpu')
    classifier.load_state_dict(torch.load(model_weights, map_location=torch.device('cpu'))['model_state_dict'])
    #classifier = classifier.backbone
    classifier.eval()
    with torch.no_grad():
        raw_image = tv.io.read_image(image_file_name).to('cpu')
        image = tv.transforms.functional.convert_image_dtype(raw_image, dtype=torch.float32)
        plt.imshow(segment_image(image, classifier).permute(1, 2, 0))
        plt.show()
        #classes = classifier(image.unsqueeze(0))['out']
        #classes = torch.nn.functional.softmax(classes[0], dim=0)
        #classes = torch.tanh(classes[0])
        #classes = torch.argmax(classes, dim=0)
        #length, width = classes.shape
        #result = torch.zeros([len(gtaloader.CLASS_COLORS), length, width], dtype=torch.bool)
        #for j in range(length):
        #    for i in range(width):
        #        result[classes[j, i], j, i] = True
        #print(result.shape)
        #print(raw_image.shape)
        #fig, axarr = plt.subplots(3, 3)
        #act_num = 0
        #for i in range(3):
        #    for j in range(3):
        #        c = classes[act_num * int((len(classes) - 1) / 9)]
                #c = classes[act_num * 16]
        #        axarr[i, j].imshow(c)
        #        act_num += 1
        #fig.savefig(output_file)
        # dont work, complains about dimensions not matching
        #output = tv.utils.draw_segmentation_masks(raw_image, result, colors = gtaloader.CLASS_COLORS)
        #output = tv.transforms.functional.convert_image_dtype(output, dtype=torch.float32)
        #tv.utils.save_image(output, output_file, format='png')
    print('Image generated')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create image')
    parser.add_argument('--model', dest='load_file_name')
    parser.add_argument('--image', dest='image_file')
    parser.add_argument('-o', '--output', dest='destination_file')

    args = parser.parse_args()
    start(args.load_file_name, args.image_file, args.destination_file)
