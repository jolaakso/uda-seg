import pytorch
import torchmetrics

def mIoU(predictions, batch_labels, label_set):
    predicted_labels = predictions.argmax(dim=1)
    matches = predicted_labels == batch_labels
    values = []
    for label in label_set:
        value = 0
        values.append(value)
    return
