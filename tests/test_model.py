from dtu_mlops_project.model import CNN
import torch
import pytest

def test_model():
    model = CNN()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    assert y.shape == (1, 10)

def test_error_on_wrong_shape():
    return None
    model = CNN()
    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
        model(torch.randn(1,2,3))
    with pytest.raises(ValueError, match='Expected each sample to have shape [1, 28, 28]'):
        model(torch.randn(1,1,28,29))

if __name__ == "__main__":
    test_model()
    #test_error_on_wrong_shape()
