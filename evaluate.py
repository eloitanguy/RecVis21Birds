from tqdm import tqdm
import os
import PIL.Image as Image
import torch
from data import data_transforms
import timm


input_size = 384

model = timm.create_model('vit_large_patch16_384', pretrained=True, num_classes=20)
checkpoint = torch.load('vit9/ep_14_V95.pth')
model.load_state_dict(checkpoint)
model.eval().cuda()


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


with torch.no_grad():
    test_dir = 'bird_dataset/test_images/mistery_category'
    output_file = open('kaggle.csv', "w")
    output_file.write("Id,Category\n")
    dt = data_transforms(input_size=input_size)
    for f in tqdm(os.listdir(test_dir)):
        if 'jpg' in f:
            data = dt(pil_loader(test_dir + '/' + f))
            data = data.view(1, data.size(0), data.size(1), data.size(2)).cuda()
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            output_file.write("%s,%d\n" % (f[:-4], pred))

    output_file.close()

print("Succesfully wrote the kaggle file, you can upload this file to the kaggle competition website")



