config = {
    'dataset': 'echr', # choices=['arxiv', 'drug', 'huffpost', 'mimic', 'fmow', 'yearbook']
    'method': 'erm', # choices=['er', 'coral', 'ensemble', 'ewc', 'ft', 'groupdro', 'irm', 'si', 'erm', 'simclr', 'swav', 'swa']
    'device': 2,  # 'gpu id'
    'random_seed': 1,  # 'random seed number'

    'eval_fix': False,

    # Training hyperparameters
    'stride': 512,
    'train_update_iter': 116000,  # 'train update iter'
    'lr': 2e-05,  # 'the base learning rate of the generator'
    'momentum': 0.9,  # 'momentum'
    'weight_decay': 0.01,  # 'weight decay
    'mini_batch_size': 2,  # 'mini batch size for SGD'
    'reduced_train_prop': None,  # 'proportion of samples allocated to train at each time step'
    'reduction': 'mean',
    'eval_freq': 5800,
    'patience': 3,

    # Evaluation
    'offline': False,  # help='evaluate offline at a single time step split'
    'difficulty': False,  # 'task difficulty'
    # todo: set value of split_time
    'split_time': 2017,  # 'timestep to split ID vs OOD'
    'test_time': 2019,
    'eval_next_timestamps': 1,  # 'number of future timesteps to evaluate on'
    'eval_worst_time': False,  # 'evaluate worst timestep accuracy'
    'load_model': False,  # 'load trained model for evaluation only'
    'eval_metric': 'acc',  # choices=['acc', 'f1', 'rmse']
    'eval_all_timestamps': False,  # 'evaluate at ID and OOD time steps'

    # ER
    'replay_freq': 150,  # 'number of previous timesteps to finetune on'

    # GroupDRO
    'num_groups': 3,  # 'number of windows for Invariant Learning baselines'
    'group_size': 2,  # 'window size for Invariant Learning baselines'
    'non_overlapping': False,  # 'non-overlapping time windows'

    # EWC
    'ewc_lambda': 0.5,  # help='how strong to weigh EWC-loss ("regularisation strength")'
    'gamma': 1.0,  # help='decay-term for old tasks (contribution to quadratic term)'
    'online': True,  # help='"online" (=single quadratic term) or "offline" (=quadratic term per task) EWC'
    'fisher_n': None,  # help='sample size for estimating FI-matrix (if "None", full pass over dataset)'
    'emp_FI': False,  # help='if True, use provided labels to calculate FI ("empirical FI"); else predicted labels'

    # A-GEM
    'buffer_size': 10,  # 'buffer size for A-GEM'

    # CORAL
    'coral_lambda': 0.01,  # 'how strong to weigh CORAL loss'

    # IRM
    'irm_lambda': 1.0,  # 'how strong to weigh IRM penalty loss'
    'irm_penalty_anneal_iters': 0,  # 'number of iterations after which we anneal IRM penalty loss'

    # Logging, saving, and testing options
    'data_dir': './data',  # 'directory for datasets.'
    'log_dir': './checkpoints',  # 'directory for summaries and checkpoints.'
    'results_dir': './results',  # 'directory for summaries and checkpoints.'
    'num_workers': 0  # 'number of workers in data generator'
}
from munch import DefaultMunch
configs = DefaultMunch.fromDict(config)

from wildtime import baseline_trainer
baseline_trainer.train(configs)