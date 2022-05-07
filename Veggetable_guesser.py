import numpy as np
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
import torch
import torchvision.transforms as transforms
from PIL import Image


trained_model = torch.load("./veggie_net.pt")
CATEGORIES = ["Bean", "Bitter_Gourd", "Bottle_Gourd", "Brinjal", "Broccoli", "Cabbage", "Capsicum", "Carrot", "Cauliflower", "Cucumber", "Papaya", "Potato", "Pumpkin", "Radish", "Tomato"]

def guess(image) :
    
    fig, ax = plt.subplots(1, 2)
    
    image_opened = cv2.imread(image)
    image_opened = cv2.cvtColor(image_opened, cv2.COLOR_BGR2RGB)
    img_array = np.array(image_opened)
    ax[0].imshow(img_array)                                                          # show the picture
    ax[0].set_title("initial image")
    
    img = Image.open(image)
    transform = transforms.Compose([transforms.Resize((50, 50)), transforms.ToTensor()])
    re_img = transform(img)
    s = torch.nn.Softmax(dim = 1)
    res = s(trained_model(re_img[None,:]))                                           # res is the resulting vector probabilities
    
    items = [k for k in range(15)]
    sns.histplot(ax = ax[1], x = items, weights = res[0].detach().numpy(), bins = 15)# plot the probability distribution and
    ax[1].set_title("probability distribution\n over the classes")
    print("I think this is a ", CATEGORIES[int(torch.argmax(res))])                  # the most probable result
    plt.show