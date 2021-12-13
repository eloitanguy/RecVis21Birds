import xgboost as xgb
import json
from config import XGBOOST_CONFIG
from models import AlexNetBackBone
from torchvision import datasets
from torch.utils.data import DataLoader
from data import data_transforms
import numpy as np
import torch
import os
from PIL import Image


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def build_xgb_dataset_alexnet(input_size):
    train_loader = DataLoader(
        datasets.ImageFolder('bird_dataset/train_images',
                             transform=data_transforms(input_size=input_size)),
        batch_size=64, shuffle=False, num_workers=8)

    val_loader = DataLoader(
        datasets.ImageFolder('bird_dataset/val_images',
                             transform=data_transforms(input_size=input_size)),
        batch_size=64, shuffle=False, num_workers=8)

    backbone = AlexNetBackBone().cuda().eval()
    Xt_list, Yt_list, Xv_list, Yv_list, Xk_list = [], [], [], [], []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.cuda()
            output = backbone(data)
            output = torch.flatten(output, start_dim=1).cpu().numpy()
            Xt_list.append(output)
            Yt_list.append(target.cpu().numpy())

    Xt = np.concatenate(Xt_list, axis=0)
    Yt = np.concatenate(Yt_list, axis=0)

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data = data.cuda()
            output = backbone(data)
            output = torch.flatten(output, start_dim=1).cpu().numpy()
            Xv_list.append(output)
            Yv_list.append(target.cpu().numpy())

    Xv = np.concatenate(Xv_list, axis=0)
    Yv = np.concatenate(Yv_list, axis=0)

    folder = 'xgboost_data/Alex' + str(input_size) + '/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    np.save(folder + 'Xt.npy', Xt)
    np.save(folder + 'Yt.npy', Yt)
    np.save(folder + 'Xv.npy', Xv)
    np.save(folder + 'Yv.npy', Yv)


class XGBRelationClassifier(object):
    def __init__(self, config=None, load=False):
        if config is None:
            config = XGBOOST_CONFIG
        self.experiment_name = config['experiment']

        if load:
            with open('xgb_experiment_results/' + self.experiment_name + '/config.json', 'r') as f:
                self.config = json.load(f)
        else:
            self.config = config

        self.model = xgb.XGBClassifier(objective='multi:softmax',
                                       max_depth=config['max_depth'],
                                       colsample_bytree=config['colsample_bytree'],
                                       n_estimators=config['n_estimators'],
                                       learning_rate=config['learning_rate'],
                                       verbosity=0,
                                       use_label_encoder=False,
                                       subsample=config['subsample'],
                                       reg_lambda=config['lambda'])
        if load:
            self.model.load_model('xgb_experiment_results/' + self.experiment_name + '/checkpoint.model')

        self.Xt_file = 'xgboost_data/Alex{}/Xt.npy'.format(config['input_size'])
        self.Yt_file = 'xgboost_data/Alex{}/Yt.npy'.format(config['input_size'])
        self.Xv_file = 'xgboost_data/Alex{}/Xv.npy'.format(config['input_size'])
        self.Yv_file = 'xgboost_data/Alex{}/Yv.npy'.format(config['input_size'])

        self.train_accuracy = -1
        self.val_accuracy = -1

    def train(self):
        Xt = np.load(self.Xt_file)
        Yt = np.load(self.Yt_file).astype(int)
        self.model.fit(Xt, Yt)
        train_predictions = self.model.predict(Xt)
        self.train_accuracy = (train_predictions == Yt).mean()
        self.config['TA'] = str(self.train_accuracy)
        print('Finished training! Train Accuracy: {:.2f}%'.format(self.train_accuracy * 100))

    def val(self):
        Xv = np.load(self.Xv_file)
        Yv = np.load(self.Yv_file).astype(int)
        val_predictions = self.model.predict(Xv)
        self.val_accuracy = (val_predictions == Yv).mean()
        self.config['VA'] = str(self.val_accuracy)
        print('Finished validation! Val Accuracy: {:.2f}%'.format(self.val_accuracy * 100))

    def save(self):
        folder = 'xgb_experiment_results/' + self.experiment_name + '/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.model.save_model(folder + 'checkpoint.model')
        with open(folder + 'config.json', 'w') as f:
            json.dump(self.config, f, indent=4)

    def kaggle_prediction(self):
        backbone = AlexNetBackBone().eval().cuda()
        test_dir = 'bird_dataset/test_images/mistery_category'
        dt = data_transforms(input_size=self.config['input_size'])
        output_file = open('kaggle.csv', "w")
        output_file.write("Id,Category\n")

        with torch.no_grad():
            for f in os.listdir(test_dir):
                if 'jpg' in f:
                    data = dt(pil_loader(test_dir + '/' + f))
                    data = data.view(1, data.size(0), data.size(1), data.size(2)).cuda()
                    output = backbone(data)
                    output = torch.flatten(output, start_dim=1).cpu().numpy()
                    prediction = int(self.model.predict(output)[0])
                    output_file.write("%s,%d\n" % (f[:-4], prediction))

        output_file.close()


build_xgb_dataset_alexnet(input_size=128)
X = XGBRelationClassifier()
X.train()
X.val()
X.kaggle_prediction()

