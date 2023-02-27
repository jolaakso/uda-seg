import torch
import torchvision as tv
import argparse
import gtaloader
import matplotlib.pyplot as plt
from DCANet import DCANet

classifier = tv.models.segmentation.deeplabv3_resnet50(num_classes = 35).backbone.to('cpu')

def start(model_weights, image_file_name, output_file):
    classifier.load_state_dict(torch.load(model_weights))
    classifier.eval()
    with torch.no_grad():
        raw_image = tv.io.read_image(image_file_name).to('cpu')
        image = tv.transforms.functional.convert_image_dtype(raw_image, dtype=torch.float32)
        classes = classifier(image.unsqueeze(0))['out']
        #classes = torch.nn.functional.softmax(classes[0], dim=0)
        classes = torch.tanh(classes[0])
        #classes = torch.argmax(classes, dim=0)
        #length, width = classes.shape
        #result = torch.zeros([len(gtaloader.CLASS_COLORS), length, width], dtype=torch.bool)
        #for j in range(length):
        #    for i in range(width):
        #        result[classes[j, i], j, i] = True
        #print(result.shape)
        #print(raw_image.shape)
        fig, axarr = plt.subplots(3, 3)
        act_num = 0
        for i in range(3):
            for j in range(3):
                c = classes[act_num * int((len(classes) - 1) / 9)]
                axarr[i, j].imshow(c)
                act_num += 1
        fig.savefig(output_file)
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
