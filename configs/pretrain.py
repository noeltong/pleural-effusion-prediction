from configs.default_config import get_config as get_default_config


def get_config():

    cfg = get_default_config()

    # ----------------
    # Train
    # ----------------

    training = cfg.training
    training.num_epochs = 300
    training.batch_size = 32
    training.save_ckpt_freq = 50
    training.eval_freq = 1

    # ----------------
    # Model
    # ----------------

    model = cfg.model
    model.clip_grad_norm = 1.
    model.ema = True
    model.ema_rate = 0.999
    model.ema_steps = 1
    model.moco_dim = 256
    model.moco_mlp_dim = 4096
    model.moco_t = 1.0
    model.moco_m = 0.99

    # ----------------
    # Optimization
    # ----------------

    optim = cfg.optim
    optim.optimizer = 'LARS'
    optim.schedule = 'CosineAnnealingLR'
    optim.loss = 'CharbonnierLoss'
    optim.grad_clip = 1.
    optim.initial_lr = 0.1
    optim.weight_decay = 0.0001
    optim.min_lr = 0.001 * optim.initial_lr
    optim.warmup_epochs = None

    # ----------------
    # Data
    # ----------------

    data = cfg.data
    data.path = 'data'
    data.num_workers = 2
    data.prefetch_factor = 1

    cfg.seed = 42
    cfg.distributed = True
    cfg.use_deterministic_algorithms = True
    cfg.debug = False

    return cfg