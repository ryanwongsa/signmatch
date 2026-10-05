import ignite.distributed as idist
import torch


def depth(name, num_blocks):
    if name.startswith(("pos_embed", "patch_embed")):
        return 0
    if name.startswith("blocks."):
        return int(name.split(".")[1]) + 1
    return num_blocks + 1


def get_optimizer(model, lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05, layer_decay=0.85):
    hiera = model.backbone.backbone.backbone
    num_layers = len(hiera.blocks) + 1
    no_weight_decay = set(hiera.no_weight_decay())

    groups = {}
    for name, param in hiera.named_parameters():
        if not param.requires_grad:
            continue
        decay = param.ndim > 1 and name not in no_weight_decay
        layer = depth(name, len(hiera.blocks))
        key = (layer, decay)
        if key not in groups:
            groups[key] = {
                "params": [],
                "lr_scale": layer_decay ** (num_layers - layer),
                "weight_decay": weight_decay if decay else 0.0,
            }
        groups[key]["params"].append(param)

    in_hiera = {id(param) for param in hiera.parameters()}
    heads = [
        {"params": [param], "lr_scale": 1.0, "weight_decay": weight_decay}
        for param in model.parameters()
        if param.requires_grad and id(param) not in in_hiera
    ]
    optimizer = torch.optim.AdamW(list(groups.values()) + heads, lr=lr, betas=tuple(betas))
    return idist.auto_optim(optimizer)
