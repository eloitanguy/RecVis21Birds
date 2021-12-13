## Object recognition and computer vision 2021/2022

This repository is the training scripts for my submission for the [Kaggle assignment](https://www.kaggle.com/c/mva-recvis-2021/leaderboard) as part of the course [Object Recognition](https://www.di.ens.fr/willow/teaching/recvis21/) at the Master "Mathematics, Vision Learning" (ENS Paris-Saclay). This work achieved a rank of 1st out of 164 participants.

### Assignment 3: Image classification 

#### Requirements

A python 3.7 (or above) version, preferably a new conda environment.

```bash
pip install -r requirements.txt
```

Warning: the default version of PyTorch is not compatible with all GPUs.

#### Dataset
We will be using a dataset containing 200 different classes of birds adapted from the [CUB-200-2011 dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).
Download the training/validation/test images from [here](https://www.di.ens.fr/willow/teaching/recvis18orig/assignment3/bird_dataset.zip). The test image labels are not provided.

In order to train with a new Train/Val/Test division, you will need to put together the default train and val sets 
into ```bird_dataset/all_images/```. If you want to use the default Train/Val division or perform K-fold cross-validation,
you will need to edit the last two lines.

#### Training and validating your model
Run the script `main.py` to train your model, the hyperparameters are all modifiable (or referred to) in ```config.py```.

#### Evaluating your model on the test set

Once a model is trained, the best checkpoints will be saved into ```[experiment_name]/```, modify ```evaluate.py``` line
12 (and above if using another model), then run ```evaluate.py```.

That generates a file `kaggle.csv` that you can upload to the private kaggle competition website.

#### XGB baseline

Simply run ```xgboost_classifiers.py```.

#### Acknowledgments
Adapted from Rob Fergus and Soumith Chintala https://github.com/soumith/traffic-sign-detection-homework.<br/>
Adaptation done by Gul Varol: https://github.com/gulvarol
