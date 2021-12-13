import os
import torch
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from data import data_transforms, Subset, get_weighted_sampler
from models import VGG16, RegNet, ResNet, EffNet
from config import CONFIG
import json
from sklearn.model_selection import train_test_split
from torch.nn import Identity
import timm


def init(random_split=False, RDS_it=1):
    experiment = CONFIG['experiment'] if not random_split else CONFIG['experiment'] + '/K_RDS_it_' + str(RDS_it)
    print('Starting experiment ', experiment)

    # Create experiment folder
    if not os.path.isdir(experiment):
        os.makedirs(experiment)

    # Storing hyperparameters
    with open(experiment + '/hyperparams.json', 'w') as f:
        json.dump(CONFIG, f, indent=4)

    # For storing results
    results = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': []
    }
    return experiment, results


def init_data():
    train_loader = DataLoader(
        datasets.ImageFolder('bird_dataset/train_images',
                             transform=data_transforms(input_size=CONFIG['input_size'], augment=CONFIG['augment'])),
        batch_size=CONFIG['batch_size'], shuffle=True, num_workers=8)
    train_examples = len(train_loader.dataset)

    val_loader = DataLoader(
        datasets.ImageFolder('bird_dataset/val_images',
                             transform=data_transforms(input_size=CONFIG['input_size'])),
        batch_size=CONFIG['batch_size'], shuffle=False, num_workers=8)
    val_examples = len(val_loader.dataset)
    return train_loader, train_examples, val_loader, val_examples


def init_data_random_split(seed=42):
    all_images = datasets.ImageFolder('bird_dataset/all_images')ù
    train_idx, val_idx = train_test_split(list(range(len(all_images))), test_size=0.1, random_state=seed)
    train_dataset = Subset(all_images, train_idx, data_transforms(input_size=CONFIG['input_size'],
                                                                  augment=CONFIG['augment']))
    val_dataset = Subset(all_images, val_idx, data_transforms(input_size=CONFIG['input_size']))
    sampler = get_weighted_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=False,  # shuffle False: sampling
                              num_workers=8, sampler=sampler)
    train_examples = len(train_loader.dataset)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=8)
    val_examples = len(val_loader.dataset)
    return train_loader, train_examples, val_loader, val_examples


def init_data_with_test_set():
    all_images = datasets.ImageFolder('bird_dataset/all_images')
    train_idx, val_test_idx = train_test_split(list(range(len(all_images))), test_size=0.2, random_state=2)
    val_test_dataset = Subset(all_images, val_test_idx, transform=Identity())
    val_idx, test_idx = train_test_split(list(range(len(val_test_dataset))), test_size=0.5, random_state=2)
    train_dataset = Subset(all_images, train_idx, data_transforms(input_size=CONFIG['input_size'],
                                                                  augment=CONFIG['augment']))
    sampler = get_weighted_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=False,  # shuffle False: sampling
                              num_workers=8, sampler=sampler)
    train_examples = len(train_loader.dataset)
    val_dataset = Subset(val_test_dataset, val_idx, data_transforms(input_size=CONFIG['input_size']))
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=8)
    val_examples = len(val_loader.dataset)
    test_dataset = Subset(val_test_dataset, test_idx, data_transforms(input_size=CONFIG['input_size']))
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=8)
    test_examples = len(test_loader.dataset)

    return train_loader, train_examples, val_loader, val_examples, test_loader, test_examples


def init_model():
    # Neural network and optimizer
    if CONFIG['model'].lower() == 'vgg16':
        model = VGG16(dropout=CONFIG['linear_dropout']).cuda().eval()
        print('VGG16 Base')
        # VGG16 FT
        optimizer = optim.Adam([
            {'params': model.features[28:].parameters(), 'lr': 1e-3},  # last conv
            {'params': model.fc.parameters(), 'lr': 1e-3},  # VGG16 FC
            {'params': model.fc_end.parameters()}  # Final linear to 20
        ],
            lr=CONFIG['lr'], weight_decay=CONFIG['wd'])

    elif CONFIG['model'].lower() == 'regnet':
        model = RegNet(dropout=CONFIG['linear_dropout']).cuda().eval()
        print('RegNet Base')
        # RegNet FT
        optimizer = optim.SGD([
            {'params': model.trunk_output[-3].parameters(), 'lr': 1e-3},  # pen-penultimate block -> regnet 6
            {'params': model.trunk_output[-2].parameters(), 'lr': 1e-3},  # penultimate block
            {'params': model.trunk_output[-1].parameters(), 'lr': 1e-3},  # last block
            {'params': model.fc.parameters()},  # RegNet FC -> 20
        ],
            lr=CONFIG['lr'], weight_decay=CONFIG['wd'], momentum=0.9)

    elif CONFIG['model'].lower() == 'resnet':
        model = ResNet(dropout=CONFIG['linear_dropout']).cuda().eval()
        print('ResNet Base')
        # ResNet FT
        optimizer = optim.Adam([
            {'params': model.fc.parameters()},
            {'params': model.layer4.parameters(), 'lr': 1e-3},
        ],
            lr=CONFIG['lr'], weight_decay=CONFIG['wd'])  #, momentum=0.9)

    elif CONFIG['model'].lower() == 'effnet':
        model = EffNet(dropout=CONFIG['linear_dropout']).cuda().eval()
        print('EffNet Base')
        # EffNet FT
        optimizer = optim.Adam([
            {'params': model.fc.parameters()},  # classifier into 20
            # {'params': model.features[-2][-1].parameters(), 'lr': 1e-3},  # last sub-block of final block
            {'params': model.features[-1].parameters(), 'lr': 1e-3}  # final conv layer
        ],
            lr=CONFIG['lr'], weight_decay=CONFIG['wd'])  #, momentum=0.9)

    else:  # CONFIG['model'].lower() == 'vit_...'
        model = timm.create_model('vit_large_patch16_384', pretrained=True, num_classes=20,
                                  drop_rate=CONFIG['linear_dropout']).cuda()
        print('ViT Base')
        optimizer = optim.SGD([
            {'params': model.blocks[-3].parameters(), 'lr': 1e-3},  # vit7
            {'params': model.blocks[-2].parameters(), 'lr': 1e-3},  # penultimate Attention block
            {'params': model.blocks[-1].parameters(), 'lr': 1e-3},  # final Attention block
            {'params': model.head.parameters()},  # final linear layer
        ],
            lr=CONFIG['lr'], weight_decay=CONFIG['wd'], momentum=0.9)

    if CONFIG['scheduler']['name'] == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=CONFIG['scheduler']['gamma'])
    else:  # CONFIG['scheduler']['name'] == 'cosWR':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                   T_0=CONFIG['scheduler']['T0'],
                                                                   T_mult=CONFIG['scheduler']['T_mult'])
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    return model, optimizer, scheduler, criterion


def validation(model, val_loader, val_examples, criterion):
    model.eval()
    with torch.no_grad():
        correct = 0
        avg_val_loss = 0
        for data, target in val_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            # sum up batch loss, account for batch size disparity (only 123 examples...)
            avg_val_loss += criterion(output, target).data.item() * len(data) / val_examples
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        accuracy = 100. * correct / val_examples
        return avg_val_loss, accuracy


def train(experiment, results, model, optimizer, scheduler, criterion,
          train_loader, train_examples, val_loader, val_examples, test_loader=None, test_examples=None):
    best_val_acc = 0
    mean_val_acc = 0
    for epoch in range(1, CONFIG['epochs'] + 1):
        model.train()
        avg_epoch_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            avg_epoch_loss += loss.item() / len(train_loader)  # more batches: assume same size: divide by n_batches
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            loss.backward()
            optimizer.step()

        train_accuracy = 100. * correct / train_examples
        print('Train Epoch: {}/{} \tEpoch AVG Loss: {:.6f}\t Accuracy: {:.1f}%\tMain LR: {:.6f}'.format(
            epoch, CONFIG['epochs'], avg_epoch_loss, train_accuracy, scheduler.get_last_lr()[0]))

        scheduler.step()
        if epoch % 1 == 0:
            val_loss, val_accuracy = validation(model, val_loader, val_examples, criterion)
            results['epochs'].append(epoch)
            results['train_loss'].append(str(avg_epoch_loss))
            results['train_accuracy'].append(str(train_accuracy))
            results['val_loss'].append(str(val_loss))
            results['val_accuracy'].append(str(val_accuracy))
            mean_val_acc = (mean_val_acc * (epoch - 1) + val_accuracy) / epoch

            if val_accuracy >= best_val_acc:
                best_val_acc = val_accuracy
                torch.save(model.state_dict(),
                           experiment + '/ep_' + str(epoch) + '_V' + str(int(best_val_acc)) + '.pth')
                if test_loader is not None:
                    test_loss, test_accuracy = validation(model, test_loader, test_examples, criterion)
                    print('Test set:\tAVG loss: {:.4f}\tAccuracy: {:.1f}%, Best VAL={:.1f}%, AVG VAL={:.1f}'.format(
                        test_loss, test_accuracy, best_val_acc, mean_val_acc))

            print('Validation set:\tAVG loss: {:.4f}\tAccuracy: {:.1f}%, Best={:.1f}%, AVG={:.1f}\n'.format(
                val_loss, val_accuracy, best_val_acc, mean_val_acc))

            with open(experiment + '/results.json', 'w') as f2:
                json.dump(results, f2, indent=4)

    return best_val_acc, mean_val_acc


def routine():
    if CONFIG['RDS_its'] == 1:
        experiment, results = init()
        train_loader, train_examples, val_loader, val_examples = init_data()
        model, optimizer, scheduler, criterion = init_model()
        train(experiment, results, model, optimizer, scheduler, criterion,
              train_loader, train_examples, val_loader, val_examples)

    else:  # expecting random splits, CONFIG['RDS_its'] > 1
        best_val_accs, mean_val_accs = [], []

        for RDS_it in range(1, CONFIG['RDS_its'] + 1):
            print('*******************************')
            print('***** RDS iteration {}/{} *****'.format(RDS_it, CONFIG['RDS_its']))
            print('*******************************')

            experiment, results = init(random_split=True, RDS_it=RDS_it)
            train_loader, train_examples, val_loader, val_examples = init_data_random_split(seed=RDS_it)
            model, optimizer, scheduler, criterion = init_model()
            best_val_acc, mean_val_acc = train(experiment, results, model, optimizer, scheduler, criterion,
                                               train_loader, train_examples, val_loader, val_examples)
            best_val_accs.append(best_val_acc)
            mean_val_accs.append(mean_val_acc)

        print('Best Accuracies: ', best_val_accs)
        rds_best = sum(best_val_accs) / len(best_val_accs)
        print('RDS Best Accuracy ', rds_best)
        print('Average Accuracies: ', mean_val_accs)
        rds_avg = sum(mean_val_accs) / len(mean_val_accs)
        print('RDS Average Accuracy ', rds_avg)

        with open(CONFIG['experiment'] + '/RDS_results.json', 'w') as f:
            json.dump({'best_val_accs': best_val_accs, 'rds_best': rds_best,
                       'mean_val_accs': mean_val_accs, rds_avg: 'rds_avg'},
                      f, indent=4)


def routine_with_test():
    experiment, results = init()
    train_loader, train_examples, val_loader, val_examples, test_loader, test_examples = init_data_with_test_set()
    model, optimizer, scheduler, criterion = init_model()
    train(experiment, results, model, optimizer, scheduler, criterion,
          train_loader, train_examples, val_loader, val_examples, test_loader=test_loader, test_examples=test_examples)


routine_with_test()  # for a Train/Val/Test division
# routine()  # for the basic Train/Val (if CONFIG['RDS_it'] = 1), or for K-fold cross val, if CONFIG['RDS_it'] = K
