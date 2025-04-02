from custom_loss import compute_custom_loss  # Add this import
import torch


def train(model, data, s_idx, c_idx, rank_matrix,
          lr=1e-3, epochs=10, opt_type='AdamW',
          lambda_pred=1.0, lambda_efk=1.0, lambda_wlr=1.0,
          lambda_avg_s=0.5, lambda_avg_c=0.5):

    if opt_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_type == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    elif opt_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown opt_type: {opt_type}")

    model.train()
    for epoch in range(epochs):
        print(f"Starting Epoch {epoch+1}/{epochs}...")  # checking if it works
        optimizer.zero_grad()
        scores = model(data)
        print("Forward pass complete.")  # checking if it works
        loss = compute_custom_loss(
            scores, data.edge_index, s_idx, c_idx, rank_matrix,
            lambda_pred=lambda_pred,
            lambda_wlr=lambda_wlr,
            lambda_avg_s=lambda_avg_s,
            lambda_avg_c=lambda_avg_c
        )
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

