from torch_geometric.data import DataLoader
import torch
from model import model 
dataset = ...  # Your graph dataset
loader = DataLoader(dataset, batch_size=32, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()  # For binary classification

for epoch in range(100):
    model.train()
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()