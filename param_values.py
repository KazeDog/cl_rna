from utils import checkattr

# def set_method_options(args, **kwargs):

def set_method_options(args, **kwargs):
    if checkattr(args, 'si'):
        args.method = 'si'
    elif checkattr(args, 'lwf'):
        args.method = 'lwf'
    elif checkattr(args, 'er'):
        args.method = 'er'
        args.replay = 'buffer'
    elif checkattr(args, 'dgr'):
        args.method = 'dgr'
        args.replay = 'generative'
    elif checkattr(args, 'icarl'):
        args.method = 'icarl'
    elif checkattr(args, 'joint'):
        args.method = 'joint'
    else:
        args.method = 'none'

# store = "./store"
store = 'store'
def set_default_values(args, also_hyper_params=True, single_context=False, no_boundaries=False):
    # -set default-values for certain arguments based on chosen experiment
    args.scenario = 'class' if args.scenario is None else args.scenario
    args.neg_samples = 'all-so-far' if args.neg_samples is None else args.neg_samples
    args.contexts = 10 if args.contexts is None else args.contexts
    args.classes = 11 if args.classes is None else args.classes
    args.seqlen = 41 if args.seqlen is None else args.seqlen
    args.step_interval = 100 if args.step_interval is None else args.step_interval
    args.data_dir = './store/datasets/' if args.data_dir is None else args.data_dir
    args.r_dir = '{}/results'.format(store) if args.r_dir is None else args.r_dir
    args.save = False if args.save is None else args.save

    args.verbose = True if args.verbose is None else args.verbose

    args.lr = 0.0001 if args.lr is None else args.lr
    args.max_epoch_num = 20 if args.max_epoch_num is None else args.max_epoch_num
    args.min_epoch_num = 10 if args.min_epoch_num is None else args.min_epoch_num
    args.batch_size = 128 if args.batch_size is None else args.batch_size

    args.z_dim = 90 if args.z_dim is None else args.z_dim
    args.gen_epoch = 100 if args.gen_epoch is None else args.gen_epoch
    args.disc_epoch = 5 if args.disc_epoch is None else args.disc_epoch

    # -gating based on internal context (brain-inspired replay)
    if args.scenario == 'task' and hasattr(args, 'dg_prop'):
        args.dg_prop = 0.
    elif args.scenario == 'domain' and hasattr(args, 'dg_prop'):
        args.dg_prop = 0.5
    elif args.scenario == 'class' and hasattr(args, 'dg_prop'):
        args.dg_prop = 0.7

    if hasattr(args, 'budget'):
        args.budget = 0.05 if args.budget is None else args.budget





