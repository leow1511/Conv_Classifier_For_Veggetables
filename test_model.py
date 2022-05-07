import torch
import seaborn as sns
from torch.utils.data import DataLoader
from data_parser import data_parser
from class_definition import VeggiesDataset, ConvNet


trained_model = torch.load("./veggie_net.pt")
# run parser only first time
data_parser("saved", "test")

test_images = torch.load(".\saved_test_values.pt")
test_labels = torch.load(".\saved_test_labels.pt")
data_test = VeggiesDataset(test_images, test_labels)
test_loader = DataLoader(data_test, batch_size = 1)
good_predictions = torch.zeros(15)
bad_predictions = torch.zeros(15)
percentage = torch.zeros(15)

for i, data in enumerate(test_loader, 0):

    inputs, labels = data["image"], data["label"]

    outputs = trained_model(inputs)
    predicted_cat = int(torch.argmax(outputs))
    real_cat = int(torch.argmax(labels))
    if predicted_cat == real_cat :
        good_predictions[real_cat] += 1
    else :
        bad_predictions[real_cat] += 1

for k in range(15) :
    percentage[k] = good_predictions[k]/(good_predictions[k] + bad_predictions[k])
    
items = [i for i in range(15)]
sns.histplot(x = items, weights = percentage.detach().numpy(), bins = 15)
percentage