import torch
import torch.nn as nn
import torch.optim as optim

# Dummy data
data = torch.rand(20, 10)  # 20 samples, 10 features each
labels = torch.randint(0, 2, (20,))  # 20 binary labels (0 or 1)

# Simple model
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1),
    nn.Sigmoid()
)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(3):  # 3 epochs
    optimizer.zero_grad()
    outputs = model(data).squeeze()
    loss = criterion(outputs, labels.float())
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
