import torch
import typer
#from dtu_mlops_project.data import corrupt_mnist
from torch_geometric.datasets import Planetoid
from dtu_mlops_project.model import GCN
import typer
from typing import Annotated
import torch_geometric

app = typer.Typer()

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

@app.command()
def evaluate(
    model_checkpoint: Annotated[str, typer.Option("--model", "-m")] = "models/model.pth"
) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(model_checkpoint)

    model = GCN().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))

    dataset = Planetoid(root="data", name="Cora")

    test_dataloader = torch_geometric.loader.DataLoader(dataset, batch_size=32, shuffle=True)

    model.eval()
    correct, total = 0, 0
    for data in test_dataloader:
        data = data.to(DEVICE)
        with torch.no_grad():
            y_pred = model(data.x, data.edge_index)
        correct += (y_pred[data.test_mask].argmax(dim=1) == data.y[data.test_mask]).sum().item()
        total += data.test_mask.sum().item()

    #for img, target in test_dataloader:
    #    img, target = img.to(DEVICE), target.to(DEVICE)
    #    with torch.no_grad():
    #        y_pred = model(img)
    #    correct += (y_pred.argmax(1) == target).float().sum().item()
    #    total += target.size(0)

    print(f"Test Accuracy: {correct / total:.4f}")

if __name__ == "__main__":
    app()
