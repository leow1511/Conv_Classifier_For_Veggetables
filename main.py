import torch
from torch.utils.data import DataLoader
from data_parser import data_parser
from class_definition import VeggiesDataset, ConvNet
from train import train

# run parser only first time
data_parser("saved", "train")

images = torch.load(".\saved_train_values.pt") # get data
labels = torch.load(".\saved_train_labels.pt")

# fit it in a loader
data_train = VeggiesDataset(images, labels)
train_loader = DataLoader(data_train, batch_size = 64, shuffle = True)

# create our net
veggie_net = ConvNet()

# start training
train(veggie_net, train_loader)

# save the model
torch.save(veggie_net, "veggie_net.pt")



