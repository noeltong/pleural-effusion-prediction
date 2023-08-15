from ml_collections.config_dict import ConfigDict
from configs.default_config import get_config as get_default_config


def get_config():

    cfg = get_default_config()

    # ----------------
    # Train
    # ----------------

    training = cfg.training
    training.num_epochs = 50
    training.batch_size = 32
    training.save_ckpt_freq = 50
    training.eval_freq = 1

    # ----------------
    # Model
    # ----------------

    model = cfg.model
    model.arch = 'resnet18'
    # model.arch = 'fasternet'
    model.clip_grad_norm = 1.
    model.ema = False
    model.ema_rate = 0.99
    model.ema_steps = 1

    # ----------------
    # Optimization
    # ----------------

    optim = cfg.optim
    optim.optimizer = 'RAdam'
    optim.schedule = 'CosineAnnealingLR'
    optim.loss = 'CharbonnierLoss'
    optim.grad_clip = 1.
    optim.initial_lr = 0.00001
    optim.weight_decay = 0.0001
    optim.min_lr = 0.001 * optim.initial_lr
    optim.warmup_epochs = None

    # ----------------
    # Data
    # ----------------

    data = cfg.data
    data.path = 'data'
    data.num_workers = 2
    data.prefetch_factor = 2

    cfg.seed = 42
    cfg.distributed = True
    cfg.use_deterministic_algorithms = True
    cfg.debug = False

    return cfg