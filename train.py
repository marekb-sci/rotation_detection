# -*- coding: utf-8 -*-
import pathlib
import numpy as np
from pylab import plt

import torch
import torchvision
from torch import nn

from data import get_dataset

class Config:
    deg_range = 60
    batch_size = 20
    num_epochs = 20
    trainset_size = 2400
    testset_size = 200
    output_dir = pathlib.Path('output_data')
    input_weights_path = None

Config.output_dir.mkdir(parents=True, exist_ok=True)

# %% data

dataset = get_dataset(radian_out=True, scale=1, random_crop=False, deg_range=Config.deg_range)
trainset, testset, _ = torch.utils.data.random_split(dataset, [Config.trainset_size, Config.testset_size, len(dataset)-Config.trainset_size-Config.testset_size])

trainloader = torch.utils.data.DataLoader(trainset, batch_size = Config.batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size = Config.batch_size)


# %% model
model = torchvision.models.densenet121(pretrained=True)
model.classifier = nn.Sequential(
    nn.Linear(model.classifier.in_features, 1),
    nn.Flatten(1, 1)
    )
if Config.input_weights_path is not None:
    model.load_state_dict(torch.load(Config.input_weights_path))

model = model.train()

# %% prepare train

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)


#%% training and evaluation function

def MSE(gt, preds):
    return np.mean((np.array(preds)-np.array(gt))**2)*180/np.pi

def MAE(gt, preds):
    return np.mean(abs(np.array(preds)-np.array(gt)))*180/np.pi


def train_one_epoch(dataloader, model, device, epoch='?'):
    history = {'gt': [], 'preds': [], 'loss': []}
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        history['gt'].extend(labels.detach().cpu().numpy())
        history['preds'].extend(outputs.detach().cpu().numpy())
        history['loss'].append(loss.detach().cpu().numpy())

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:    # print every 10 mini-batches
            print(f'epoch: {epoch},\timage: {(i+1)*dataloader.batch_size}/{len(dataloader.dataset)},\tloss: {running_loss / 10:0.3f}')
            running_loss = 0.0

    mse = MSE(history['gt'], history['preds'])
    mae = MAE(history['gt'], history['preds'])
    mean_loss = np.mean(history['loss'])
    return {'mse': mse, 'mae': mae, 'loss': mean_loss}

def evaluate(dataloader, model, device, plot_results=False):
    gt = []
    preds = []

    for i, data in enumerate(dataloader, 0):

        # get the inputs; data is a list of [inputs, labels]
        with torch.no_grad():
            inputs, labels = data
            outputs = model(inputs.to(device)).squeeze().cpu()

        gt.extend(list(labels.numpy()))
        preds.extend(list(outputs.numpy()))

    gt = np.array(gt)
    preds = np.array(preds)
    print(f'testset evaluation completed, mean square error: {MSE(gt,preds)/np.pi:0.3f}, mean abs: {MAE(gt,preds):0.3f}\n')
    if plot_results:
        plt.figure()
        plt.hist(np.degrees(preds-gt), bins=20, density=True, alpha=0.7)
    return gt, preds


# %% training
history = []
best_test_mse = np.inf
for epoch in range(Config.num_epochs):  # loop over the dataset multiple times
    print(f'EPOCH {epoch}')

    train_results = train_one_epoch(trainloader, model.train(), device, epoch)
    eval_gt, eval_preds = evaluate(testloader, model.eval(), device)
    test_results = {'mse': MSE(eval_gt, eval_preds), 'mae': MAE(eval_gt, eval_preds) }
    history.append({'train': train_results, 'test': test_results})

    if test_results['mse'] < best_test_mse:
        best_test_mse = test_results['mse']
        torch.save(model, Config.output_dir / 'model_weights_best.pt')

print('Finished Training')
## % save
import pickle

with open(Config.output_dir / 'history.json', 'wb') as f:
    pickle.dump(history, f)

#%% final evaluation and visualisation

model = torch.load(Config.output_dir / 'model_weights_best.pt').eval().to(device)

print('\nFINAL EVALUATION:')
gt, preds = evaluate(trainloader, model, device)
mse = MSE(gt, preds)
print(f'train dataset: {mse}')
_, bins, _ = plt.hist(np.degrees(preds-gt), bins=20, density=True, alpha=0.7, label=f'train, mse: {mse:0.2f}')

gt, preds = evaluate(testloader, model, device)
mse = MSE(gt, preds)
print(f'test dataset: {mse}')
plt.hist(np.degrees(preds-gt), bins=bins, density=True, alpha=0.7, label=f'train, mse: {mse:0.2f}')