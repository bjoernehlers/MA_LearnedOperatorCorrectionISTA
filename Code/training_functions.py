#training and validation loops
import torch
import numpy as np
import matplotlib.pyplot as plt

def train_loop(data_loader, model, loss_fn, optimizer, device, show_step=0.2):
    model.train()
    size = len(data_loader.dataset)
    k = show_step
    mean_loss = 0
    max_loss = 0   
    for batch, (input_data, true_output_data) in enumerate(data_loader):
        in_data = torch.unsqueeze(input_data, 1)
        in_data = in_data.to(device).float()
        goal_data = torch.unsqueeze(true_output_data, 1)
        goal_data = goal_data.to(device).float()

        # Forward pass
        output_data = model(in_data).float()
        loss = loss_fn(output_data, goal_data)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss = loss.item()
        mean_loss = mean_loss + loss
        if loss > max_loss:
            max_loss = loss
        if ((batch+1)/size) >= k:
            k = k + show_step
            print(f"loss: {loss:>7f} [{batch:>5d}/{size:>5d}]")
    mean_loss = mean_loss/size
    print(f"mean train loss: {mean_loss} | max train loss: {max_loss}")
    return mean_loss

def val_loop(data_loader, model, loss_fn,device):
    model.eval()
    size = len(data_loader.dataset)
    mean_loss = 0
    max_loss = 0
    with torch.no_grad():
        for batch, (input_data, true_output_data) in enumerate(data_loader):
            in_data = torch.unsqueeze(input_data, 1)
            in_data = in_data.to(device).float()
            goal_data = torch.unsqueeze(true_output_data, 1)
            goal_data = goal_data.to(device).float()

            # Forward pass
            output_data = model(in_data)
            loss = loss_fn(output_data, goal_data)
            mean_loss = mean_loss + loss.item()
            if loss.item() > max_loss:
                max_loss = loss.item()
    mean_loss = mean_loss/size
    print(f"mean val loss: {mean_loss} | max val loss: {max_loss}")
    return mean_loss