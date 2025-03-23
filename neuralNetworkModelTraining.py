import torch
import torch.nn as nn
from neuralNetworkModel import NeuralNetworkModel
from data import DigitsData
import matplotlib.pyplot as plt
import numpy as np
#define device
device = torch.device('cuda')
#Hyper-Parameters
input_size = 28 * 28
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.0001
regularization = 0.001

img_loader = DigitsData(batch_size, 9)

model = NeuralNetworkModel(input_size, hidden_size, num_classes).to(device)

lossFunc = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate, weight_decay=regularization)

losses = []
for epoch in range(num_epochs):
    #RUN FOR EACH EPOCH
    for i, (images, labels) in enumerate(img_loader.train_dataloader):
        #LOOP FOR EACH BATCH
        images = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)
        yhats = model(images)
        loss = lossFunc(yhats, labels)

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        if i % (images.shape[0]/5) == 0:
            losses.append(loss)

        print(f"EPOCH: {epoch}, STEP: {i}, LOSS: {loss}")

print(f"END OF TRAINING..")

#DISPLAY A GRID 
with torch.no_grad():
    test_data, test_labels = next(iter(img_loader.test_dataloader))
    test_data = test_data.reshape(-1, input_size).to(device)
    test_data_predicted_labels = model(test_data)

    test_data_predicted_labels = test_data_predicted_labels.argmax(dim=1)

    test_data_predicted_labels =test_data_predicted_labels.cpu()
    test_data = test_data.cpu()
    size =  int(np.sqrt(9)) 
    fig, ax = plt.subplots(size, size)
    
    print(test_data.shape)
    for y in range(size):
        for x in range(size):
            i = (y * (size) + x)
            ax[y,x].imshow(test_data[i].reshape(28,28), cmap='gray')
            ax[y,x].set_title(f"{test_labels[i]}, {test_data_predicted_labels[i]}")
            ax[y,x].set_axis_off()

    plt.tight_layout()
    plt.show()



