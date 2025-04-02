import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD, RMSprop, AdamW

def train(model, data, s_idx, c_idx, rank_matrix, lr=1e-3, epochs=10, opt_type='Adam'):
    if opt_type == 'Adam':
        optimizer = Adam(model.parameters(), lr=lr)
    elif opt_type == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    elif opt_type == 'RMSprop':
        optimizer = RMSprop(model.parameters(), lr=lr)
    elif opt_type == 'SGD':
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        scores = model(data)
        edge_index = data.edge_index

        loss = model.compute_loss(scores, rank_matrix, edge_index, s_idx, c_idx)

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
