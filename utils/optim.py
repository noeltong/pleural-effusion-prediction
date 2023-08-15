import torch
from torch import nn


class LARS(torch.optim.Optimizer):
    """
    LARS optimizer, no rate scaling or weight decay for parameters <= 1D.
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, trust_coefficient=0.001):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, trust_coefficient=trust_coefficient)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim > 1: # if not normalization gamma/beta or bias
                    dp = dp.add(p, alpha=g['weight_decay'])
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                    (g['trust_coefficient'] * param_norm / update_norm), one),
                                    one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)
                p.add_(mu, alpha=-g['lr'])



def get_optim(model, config):
    init_lr = config.optim.initial_lr * config.training.batch_size / 256

    if config.optim.optimizer.lower() == 'lars':
        optimizer = LARS(model.parameters(),
                         init_lr,
                         weight_decay=config.optim.weight_decay
        )
    elif config.optim.optimizer.lower() == 'radam':
        optimizer = torch.optim.RAdam(
            model.parameters(),
            lr=init_lr,
            weight_decay=config.optim.weight_decay,
        )
    else:
        raise NotImplementedError(
            f'{config.optim.optimizer} is not supported.')

    if config.optim.schedule.lower() is not None and config.optim.schedule.lower() == 'cosineannealinglr':
        if config.optim.warmup_epochs is None:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.training.num_epochs,
                eta_min=config.optim.min_lr,
            )
        else:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=config.optim.initial_lr / config.optim.warmup_epochs,
                total_iters=config.optim.warmup_epochs
            )
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.training.num_epochs - config.optim.warmup_epochs,
                eta_min=config.optim.min_lr
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[config.optim.warmup_epochs]
            )
    else:
        raise ValueError(f'{config.optim.schedule} is not supported.')

    return optimizer, scheduler
