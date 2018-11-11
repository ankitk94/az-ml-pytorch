import torch
from torch.autograd import Variable
import torch.nn as nn
from sklearn.datasets import load_boston
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from azureml.core.run import Run

lr = 0.01        # Learning rate
epoch_num = 1000

run = Run.get_submitted_run()

os.makedirs('./outputs', exist_ok=True)

boston = load_boston()

features = torch.from_numpy(np.array(boston.data)).float()  # convert the numpy array into torch tensor
features = Variable(features)                       # create a torch variable and transfer it into GPU

labels = torch.from_numpy(np.array(boston.target).reshape(506,1)).float()  # convert the numpy array into torch tensor
labels = Variable(labels)                            # create a torch variable and transfer it into GPU

linear_regression = nn.Linear(13, 1)

# define the loss (criterion) and create an optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(linear_regression.parameters(), lr=lr)

run = Run.get_context()


for ep in range(epoch_num):  # epochs loop
    # Reset gradients
    linear_regression.zero_grad()

    # Forward pass
    output = linear_regression(features)
    loss = criterion(output, labels)        # calculate the loss

    if not ep%500:
        print('Epoch: {} - loss: {}'.format(ep, loss.data[0]))

    # Backward pass and updates
    loss.backward()                         # calculate the gradients (backpropagation)
    optimizer.step()                        # update the weights


output = output.data.cpu().numpy()
labels = np.array(boston.target)

fig, ax = plt.subplots()
ax.scatter(labels, output)
ax.plot([labels.min(), labels.max()], [labels.min(), labels.max()], 'k--', lw=3)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
torch.save(linear_regression, "outputs/model.pth")
