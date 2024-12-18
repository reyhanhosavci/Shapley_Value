import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import csv
import medmnist
from medmnist import INFO, Evaluator
print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")

data_flag = 'pathmnist'
# data_flag = 'breastmnist'
download = True

NUM_EPOCHS = 3
BATCH_SIZE = 128
lr = 0.001

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])

# preprocessing
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# load the data
train_dataset = DataClass(split='train', transform=data_transform, download=download)
test_dataset = DataClass(split='test', transform=data_transform, download=download)

pil_dataset = DataClass(split='train', download=download)

# encapsulate data into dataloader form
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_loader_4_exp = data.DataLoader(dataset=train_dataset, batch_size=3, shuffle=False)
train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_loader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

class Net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = Net(in_channels=n_channels, num_classes=n_classes)    
# Load
model.load_state_dict(torch.load("cnnmodel.pt", weights_only=True))
model.eval()


import shap
import numpy as np

#Creating explainers 
list_explainers=[]
shapley_values=[]
shap_numpy=[]
for k in range(10): # to create 10 different deep explainer
    batch= next(iter(train_loader))
    images, target_im , idx= batch
    
    e = shap.DeepExplainer(model, images)
    list_explainers.append(e)
    e_values =[]
    for l, (test_images, test_target, test_idx) in enumerate(train_loader_4_exp): # iteration for all train values 1000 for each 
        if l==90: #loop 90 for 90k image & 1k data loader 
            break
        shapley_values=e.shap_values(test_images)
        shap_numpy=list(np.transpose(shapley_values, (4, 0, 2, 3, 1))) #class/number of images/28/28/3
       
        for i in range(np.size(shap_numpy,1)):
            col=[test_idx[i].numpy(), int(test_target[i].numpy())]
            col2=[]
            for j in range(n_classes):
                col2.append(np.sum(shap_numpy[j][i]))
            col.append(col2[int(test_target[i].numpy())])
            e_values.append(col)
    
    file_name=str(k) + ". explainer Shapley Values.csv"
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(e_values)

        