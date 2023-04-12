from ml_collections.config_dict import ConfigDict


def get_config():

    cfg = ConfigDict()

    # ----------------
    # Train
    # ----------------

    training = cfg.training = ConfigDict()
    training.num_epochs = 10000
    training.batch_size = 32
    training.save_ckpt_freq = 100
    training.eval_freq = 5

    # ----------------
    # Model
    # ----------------

    model = cfg.model = ConfigDict()
    model.clip_grad_norm = 1.
    model.ema = False
    model.ema_rate = 0.99
    model.ema_steps = 1
    model.depths = [2, 2, 2, 2]

    # ----------------
    # Optimization
    # ----------------

    cfg.optim = optim = ConfigDict()
    optim.optimizer = 'AdamW'
    optim.schedule = 'CosineAnnealingLR'
    optim.grad_clip = 1.
    optim.initial_lr = 0.000075
    optim.weight_decay = 0.0001
    optim.min_lr = 0.001 * optim.initial_lr
    optim.warmup_epochs = 250

    # ----------------
    # Data
    # ----------------

    cfg.data = data = ConfigDict()
    data.path = '/root/pred-pe/data'
    data.num_workers = 2
    data.prefetch_factor = 2

    cfg.seed = 42
    cfg.distributed = True
    cfg.use_deterministic_algorithms = True
    cfg.debug = False

    return cfg