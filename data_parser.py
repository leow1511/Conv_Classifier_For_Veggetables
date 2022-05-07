import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
import torch
import torchvision.transforms as transforms
from PIL import Image


img_size = 50
CATEGORIES = ["Bean", "Bitter_Gourd", "Bottle_Gourd", "Brinjal", "Broccoli", "Cabbage", "Capsicum", "Carrot", "Cauliflower", "Cucumber", "Papaya", "Potato", "Pumpkin", "Radish", "Tomato"]


def data_parser(output_file_name, task, CATEGORIES = CATEGORIES, img_size = img_size):
    img_data = torch.tensor([]) # the BIG tensor that will store all our image tensors
    class_data = [] # future tensor which will store our labels
    transform = transforms.Compose([transforms.Resize((50, 50)), transforms.ToTensor()]) # we resize, then convert to tensor.
    
    for category in CATEGORIES:
        class_number = CATEGORIES.index(category)
        print("parsing ", category)
        
        for picture in os.listdir('./' + task + '/' + category) :            
            img = Image.open(os.path.join('./', task, category, picture))
            img_tensor = transform(img)
            img_data = torch.cat((img_data, img_tensor[None,:]), 0)
            class_data.append(class_number)
            
    print("Converting labels to 1-Hot...") # we convert the labels with a 1-Hot encoding so it is adequate for classification
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(np.array(class_data).reshape(-1, 1))
    class_data = enc.transform(np.array(class_data).reshape(-1, 1)).toarray()
    class_data = torch.tensor(class_data)
    
    print("Saving data...")
    torch.save(img_data, output_file_name + "_" + task + "_values.pt")
    torch.save(class_data, output_file_name + "_" + task + "_labels.pt")
    return "Done."