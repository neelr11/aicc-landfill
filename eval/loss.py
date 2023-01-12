import torch
import argparse


def get_loss_fn(loss_args):
    loss_args_ = loss_args
    if isinstance(loss_args, argparse.Namespace):
        loss_args_ = vars(loss_args)
    loss_fn = loss_args_.get("loss_fn")

    pos_weight = loss_args_.get("pos_weight")
    if pos_weight:
        pos_weight = torch.tensor(pos_weight)
    
    if loss_fn == "BCE":
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_fn == "CE":
        return torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"loss_fn {loss_args.loss_fn} not supported.")
