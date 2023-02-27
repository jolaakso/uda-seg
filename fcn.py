import torchvision as tv

img = tv.io.image.read_image("english-black-lab-puppy.jpg")

weights = tv.models.segmentation.FCN_ResNet50_Weights.DEFAULT
model = tv.models.segmentation.fcn_resnet50(weights=weights)
model.eval()

preprocess = weights.transforms()

batch = preprocess(img).unsqueeze(0)

prediction = model(batch)["out"]
normalized_masks = prediction.softmax(dim=1)
class_to_idx = { cls: ix for (ix, cls) in enumerate(weights.meta["categories"])}
print(weights.meta["categories"])
mask = normalized_masks[0, class_to_idx["car"]]
tv.transforms.functional.to_pil_image(mask).show()
