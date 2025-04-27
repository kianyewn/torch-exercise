import torch
import torch.nn as nn


def get_number_of_parameters(model):
    return sum([p.numel() for p in model.parameters()])


def get_number_of_trainable_parameters(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def get_grad_norm(model, max_grad_norm=1e-9):
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_grad_norm
    )
    return grad_norm


def test_get_num_parameters():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(10, 10)

            self.dummy_parameter = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        def forward(self, x):
            return self.layer(x)

    model = Model()
    num_params = get_number_of_parameters(model)
    num_trainable_params = get_number_of_trainable_parameters(model)

    assert num_params == 111
    assert num_trainable_params == 110
