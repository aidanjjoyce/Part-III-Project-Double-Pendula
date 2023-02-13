import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from ast import literal_eval

learning_rate = 0.001
n_epochs = 2
batch = 3
n_coords = 40

# Import data from csv

df = pd.read_csv(r"C:\Users\aidan\Documents\Part III Project\Part-III-Project-Double-Pendula\10_data_100_pendula.txt", usecols=[1,2])
to_array = pd.DataFrame(df).to_numpy()
data = np.array([literal_eval(to_array[i][0]) + literal_eval(to_array[i][1]) for i in range(len(to_array))])
data = torch.from_numpy(data)
data = data.to(torch.float32)


n_coords = len(data[0]) - 1

# Defines the training and testing datasets and dataloaders

trainloader = torch.utils.data.DataLoader(data[0:round(0.7*len(data))], batch_size=batch, shuffle=True)

testloader = torch.utils.data.DataLoader(data[round(0.7*len(data)):len(data)], batch_size=batch, shuffle=False)

# Define neural network for model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(n_coords,10)
        self.l2 = nn.Linear(10,1)
        self.l3 = nn.Linear(10,1)
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return x
        
net = Net()

criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)

print("Started Training")
for epoch in range(n_epochs):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0][:-1], data[0][-1]
        
        optimiser.zero_grad()
        
        outputs = net(inputs)
        print(outputs)
        print(labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()
        
        running_loss += loss.items
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
           
print("Finished Training")

PATH
torch.save(net.state_dict(), PATH)

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        