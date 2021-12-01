# import necessary libraries

import torch
import numpy as np
import os
import model
from tqdm import tqdm
from torch import nn, optim
from numpy import linalg as LA
from scipy.stats import rankdata
from collections import OrderedDict
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
import classifier

# Function to check test accuracy
def test_accuracy(model, testloader, criterion):
    test_loss = 0.0
    class_correct = list(0. for _ in range(102))
    class_total = list(0. for _ in range(102))
    model.eval()
    for data, target in tqdm(testloader):
        # forward pass
        output = model(data)
        # Get the loss
        loss = criterion(output, target)
        # Test loss
        test_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class and get the max class
        _, pred = torch.max(output, 1)
        # compare prediction with true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # calculate test accuracy for each object class
        for i in range(len(target)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
    # Overall accuracy
    overall_accuracy = 100. * np.sum(class_correct) / np.sum(class_total)
    return overall_accuracy

if __name__ == '__main__':
    data_transforms = {
            'valid': transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ]),
        }
    data_dir = './flower_data'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['valid']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                                 shuffle=True, num_workers=0)
                  for x in ['valid']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['valid']}
    device = torch.device("cpu")
    model_ft = classifier.vgg11_bn(pretrained=False).to(device)
    model_ft.load_state_dict(torch.load("model/vgg11.pt",map_location=torch.device('cpu')))
    criterion = nn.NLLLoss()
    #acc = test_accuracy(model_ft, dataloaders['valid'], criterion)
    #print(acc)
    print(sum(p.numel() for p in model_ft.parameters() if p.requires_grad))
    # setting configure
    prune_percentage = [.0, .30]
    accuracies_np = []
    # Get the accuracy without any pruning
    #accuracies_np.append(acc)
    # Neuron Pruning
    # Code is almost same as above so comments are provided for only different parts of code
    r = [-8,-6,-4]

    #print(list(model_ft.state_dict()))
    for k in prune_percentage[1:]:
        weights = model_ft.state_dict()
        layers = list(model_ft.state_dict())
        #print(weights)
        ranks = {}
        pruned_weights = []

        for w in layers[:-8]:
            pruned_weights.append(weights[w])

        for i in tqdm(r):
            l = layers[i]
            data = weights[l]
            w = np.array(data)
            # taking norm for each neuron
            norm = LA.norm(w, axis=0)
            # repeat the norm values to get the shape similar to that of layer weights
            norm = np.tile(norm, (w.shape[0],1))
            ranks[l] = (rankdata(norm, method='dense') - 1).astype(int).reshape(norm.shape)
            lower_bound_rank = np.ceil(np.max(ranks[l])*k).astype(int)
            ranks[l][ranks[l] <= lower_bound_rank] = 0
            ranks[l][ranks[l] > lower_bound_rank] = 1
            new_w = w * ranks[l]
            data = torch.from_numpy(new_w)
            pruned_weights.append(data)
            pruned_weights.append(weights[layers[i+1]])

        pruned_weights.append(weights[layers[-2]])
        pruned_weights.append(weights[layers[-1]])
        new_state_dict = OrderedDict()
        for l, pw in zip(layers, pruned_weights):
            new_state_dict[l] = pw

        new_model = classifier.vgg11_bn(pretrained=False).to(device)
        new_model.load_state_dict(new_state_dict)
        accuracies_np.append(test_accuracy(new_model,dataloaders['valid'], criterion))

    print(sum(p.numel() for p in model_ft.parameters() if p.requires_grad))
    print(accuracies_np)