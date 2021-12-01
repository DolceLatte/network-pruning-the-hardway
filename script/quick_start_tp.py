import torch
from torchvision.models import resnet18
import torch_pruning as tp
import classifier
from torchvision import transforms,datasets
import os
from torch import nn
import numpy as np
from tqdm import tqdm

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

data_transforms = {
            'valid': transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ]),
        }
data_dir = '../flower_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['valid']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                             shuffle=True, num_workers=0)
              for x in ['valid']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['valid']}

criterion = nn.NLLLoss()

device = torch.device("cpu")
model = classifier.vgg11_bn(pretrained=False).to(device)
model.load_state_dict(torch.load("../model/vgg11.pt",map_location=torch.device('cpu')))


layers = list(model.state_dict())
print(model)
print(layers)
print(len(layers))
#acc = test_accuracy(model, dataloaders['valid'], criterion)
#print(acc)

# 1. setup strategy (L1 Norm)
strategy = tp.strategy.L2Strategy() # or tp.strategy.RandomStrategy()

# 2. build layer dependency for resnet18
DG = tp.DependencyGraph()
DG.build_dependency(model, example_inputs=torch.randn(1,3,224,224))

# 3. get a pruning plan from the dependency graph.
pruning_idxs = strategy(model.classifier[0].weight, amount=0.3)
pruning_plan = DG.get_pruning_plan( model.classifier[0], tp.prune_linear, idxs=pruning_idxs)
pruning_plan.exec()
print(sum(p.numel() for p in model.parameters() if p.requires_grad))

pruning_idxs = strategy(model.classifier[3].weight, amount=0.3)
pruning_plan = DG.get_pruning_plan( model.classifier[3], tp.prune_linear, idxs=pruning_idxs)
pruning_plan.exec()
print(sum(p.numel() for p in model.parameters() if p.requires_grad))

pruning_idxs = strategy(model.classifier[6].weight, amount=0.3)
pruning_plan = DG.get_pruning_plan( model.classifier[6], tp.prune_linear, idxs=pruning_idxs)
pruning_plan.exec()
print(sum(p.numel() for p in model.parameters() if p.requires_grad))

acc = test_accuracy(model, dataloaders['valid'], criterion)
print(acc)
