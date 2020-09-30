import torch


def Optimizer(model,**params):
    return torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                **params)
