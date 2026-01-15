from dtu_mlops_project.model import GCN
from torch_geometric.datasets import Planetoid
import torch
import torch_geometric
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig
import os

plt.style.use("ggplot")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

@hydra.main(config_path="../../configs", config_name="config", version_base="1.1")
def train(cfg: DictConfig) -> None:
    print("Training started...")
    print("Device:", DEVICE)
    print(f"lr={cfg.hyperparameters.lr}, batch_size={cfg.hyperparameters.batch_size}, epochs={cfg.hyperparameters.epochs}")
    
    # Set seed for reproducibility
    torch.manual_seed(cfg.hyperparameters.seed)
    
    # Initialize model
    model = GCN().to(DEVICE)
    
    # Get original working directory (Hydra changes it)
    original_cwd = hydra.utils.get_original_cwd()
    
    # Load dataset using absolute path
    dataset_root = os.path.join(original_cwd, cfg.dataset.root)
    dataset = Planetoid(root=dataset_root, name=cfg.dataset.name)
    
    train_dataloader = torch_geometric.loader.DataLoader(
        dataset, 
        batch_size=cfg.hyperparameters.batch_size, 
        shuffle=True
    )
    
    # Setup training
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.hyperparameters.lr)
    
    # Collect statistics
    statistics = {
        "train_loss": [],
        "train_accuracy": [],
    }
    
    for epoch in range(cfg.hyperparameters.epochs):
        model.train()
        for i, data in enumerate(train_dataloader):
            data = data.to(DEVICE)
            optimizer.zero_grad()
            
            # Forward pass
            out = model(data.x, data.edge_index)
            
            # Compute loss only for training nodes
            loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
            
            # Backpropagation and optimization step
            loss.backward()
            optimizer.step()
            
            # Log statistics
            statistics["train_loss"].append(loss.item())
            accuracy = (out[data.train_mask].argmax(dim=1) == data.y[data.train_mask]).sum().item() / data.train_mask.sum().item()
            statistics["train_accuracy"].append(accuracy)
            
            if i % 100 == 0:
                print(f"Epoch {epoch+1}, iter {i}, loss: {loss.item()}")
    
    print("Training finished.")
    
    # Save model using absolute path
    model_dir = os.path.join(original_cwd, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, cfg.output.model_path)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Plot statistics
    figures_dir = os.path.join(original_cwd, cfg.output.figures_dir)
    os.makedirs(figures_dir, exist_ok=True)
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig(os.path.join(figures_dir, "training_statistics.pdf"))
    print(f"Figures saved to {figures_dir}")

if __name__ == "__main__":
    train()