# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.optim as optim
import os
import torch
import torch.nn as nn
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import copy
#训练模型
def train_model(model, trainloader,devloader,testloader, epochs, lr, modelname):
    if not os.path.exists(os.path.join("./model", modelname)):
        os.makedirs(os.path.join("./model", modelname))
    optimizer = optim.Adam(model.parameters(), lr=lr)    #Adam优化器
    best_val_rmse = float('inf')
    #best_test_emse = float('inf')
    print('Start training')
    for epoch in range(1, epochs + 1):
        train_loss = train(model, trainloader, optimizer)
        val_rmse = evaluate(model, devloader)
        #test_rmse = evaluate(model, testloader)
        print('-' * 50)
        print(f'Epoch:{epoch:>3} | [Train] | Loss: {train_loss:>.3f}')
        print(f'Epoch:{epoch:>3} | [Val]   | MSE : {val_rmse:>.3f} ')     #输出训练和验证loss
        #print(f'Epoch:{epoch:>3} | [Test]  | MSE : {test_rmse:>.3f} ')     #输出训练和验证loss
        print('-' * 50)

        torch.save(model, os.path.join("./model", modelname, f"{epoch}.pth"))
        if val_rmse<best_val_rmse:
            best_val_rmse = val_rmse
    print(f"Best val rmse: {best_val_rmse}")

#每轮训练模型
def train(model, trainloader, optimizer):
    model.train()
    running_loss = 0.
    criterion = torch.nn.MSELoss()

    for i, train_data in enumerate(trainloader):

        features, labels, lengths = train_data

        optimizer.zero_grad()
        features, labels = features.cuda(), labels.cuda()
        predictions = model.forward(features,lengths)
        loss = criterion(predictions, labels.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(trainloader)


def evaluate(model, val_loader):
    model.eval()
    with torch.no_grad():
        predictions_corr = np.empty((0, 8))
        labels_corr = np.empty((0, 8))

        for i, val_data in enumerate(val_loader):
            if len(val_data)==3:
                features, labels, lengths = val_data
            else:
                features,labels = val_data

            features = features.cuda()

            predictions = model.forward(features,lengths)
            predictions = predictions.cpu().numpy()
            labels = labels.cpu().numpy()

            predictions_corr = np.append(predictions_corr, predictions, axis=0)
            labels_corr = np.append(labels_corr, labels, axis=0)

        labels_corr = labels_corr
        predictions_corr = predictions_corr
        rmse = sqrt(mean_squared_error(predictions_corr, labels_corr))

    return rmse