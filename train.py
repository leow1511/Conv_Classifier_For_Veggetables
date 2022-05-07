import torch.nn as nn
import torch.optim as optim

epochs = 50
criterion = nn.CrossEntropyLoss()

def train(model, train_loader, epochs = epochs) :
    
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(epochs): 

        count = 0
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):

            inputs, labels = data["image"], data["label"]
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            count += 1

        print("mean epoch loss for epoch ", epoch + 1, " is ", running_loss/count)

    print('Finished Training')