import argparse

##-------------------------------------------------------------------------------------------------------------------##

# Where to store the data / results / models / plots
store = "./store"


##-------------------------------------------------------------------------------------------------------------------##

####################
## Define options ##
####################

def define_args(filename, description):
    parser = argparse.ArgumentParser('./{}.py'.format(filename), description=description)
    return parser


def add_general_options(parser, main=False, comparison=False, compare_hyper=False, pretrain=False, **kwargs):
    parser.add_argument('--seed', type=int, default=0, help='[first] random seed (for each random-module used)')
    parser.add_argument('--no-gpus', action='store_false', dest='cuda', help="don't use GPUs")

    parser.add_argument('--plot-dir', type=str, default='{}/plots'.format(store), dest='p_dir',
                        help="default: %(default)s")
    parser.add_argument('--results-dir', type=str, dest='r_dir',
                        help="default: %(default)s")

    parser.add_argument('--test', action='store_false', dest='train', help='evaluate previously saved model')

    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--index', type=int, default=0)

    return parser


##-------------------------------------------------------------------------------------------------------------------##

def add_eval_options(parser, main=False, comparison=False, pretrain=False, compare_replay=False, no_boundaries=False,
                     **kwargs):
    eval_params = parser.add_argument_group('Evaluation Parameters')

    eval_params.add_argument('--time', action='store_true', help="keep track of total training time")
    eval_params.add_argument('--test_size', type=int, default=None,
                             help="# samples for evaluating accuracy (for visdom)")

    return parser


def add_problem_options(parser, pretrain=False, no_boundaries=False, **kwargs):
    problem_params = parser.add_argument_group('Problem Specification')

    problem_params.add_argument('--scenario', type=str, choices=['task', 'domain', 'class'])
    problem_params.add_argument('--contexts', type=int, metavar='N', help='number of contexts')
    problem_params.add_argument('--classes', type=int, help='number of class')
    problem_params.add_argument('--seqlen', type=int, help='the length of the intercepted sequence')
    # problem_params.add_argument('--epoch', type=int, help="training epoch")
    problem_params.add_argument('--batch_size', type=int, help="mini batch size (# observations per iteration)")
    problem_params.add_argument('--method', type=str)

    return parser


def add_model_options(parser, pretrain=False, compare_replay=False, **kwargs):
    model = parser.add_argument_group('Parameters Main Model')

    # -fully connected layers
    model.add_argument('--fc-layers', type=int, default=3, dest='fc_lay', help="# of fully-connected layers")
    model.add_argument('--fc-units', type=int, metavar="N", help="# of units in hidden fc-layers")
    model.add_argument('--fc-drop', type=float, default=0., help="dropout probability for fc-units")
    model.add_argument('--fc-bn', type=str, default="no", help="use batch-norm in the fc-layers (no|yes)")
    model.add_argument('--fc-nl', type=str, default="relu", choices=["relu", "leakyrelu", "none"])

    model.add_argument('--singlehead', action='store_true',
                       help="for Task-IL: use a 'single-headed' output layer  (instead of a 'multi-headed' one)")

    return parser


def add_train_options(parser, main=False, no_boundaries=False, pretrain=False, compare_replay=False, **kwargs):
    ## Training hyperparameters
    train_params = parser.add_argument_group('Training Parameters')

    train_params.add_argument('--lr', type=float, help="learning rate")
    train_params.add_argument('--data_dir', type=str)
    train_params.add_argument('--optimizer', type=str, default='adam',
                              choices=['adam', 'sgd'] if no_boundaries else ['adam', 'adam_reset', 'sgd'])
    train_params.add_argument("--max_epoch_num", action="store", type=int,
                              required=False, help="max epoch num, default 10")
    train_params.add_argument("--min_epoch_num", action="store", type=int,
                              required=False, help="min epoch num, default 5")
    train_params.add_argument('--step_interval', type=int, required=False)
    train_params.add_argument('--active-classes', type=str,
                              choices=["all", "all-so-far", "current"],
                              dest='neg_samples', help="for Class-IL: which classes to set to 'active'?")
    if (not pretrain) and (not compare_replay):
        loss_params = parser.add_argument_group('Loss Parameters')
        loss_params.add_argument('--recon-loss', type=str, choices=['MSE', 'BCE'])
    if main:
        loss_params.add_argument('--bce', action='store_true',
                                 help="use binary (instead of multi-class) classification loss")
    if main and (not no_boundaries):
        loss_params.add_argument('--bce-distill', action='store_true', help='distilled loss on previous classes for new'
                                                                            ' examples (if --bce & --scenario="class")')

    return parser


def add_cl_options(parser, main=False, compare_all=False, compare_replay=False, compare_hyper=False,
                   no_boundaries=False, **kwargs):

    if main and (not no_boundaries):
        baseline_options = parser.add_argument_group('Baseline Options')
        baseline_options.add_argument('--joint', action='store_true', help="train once on data of all contexts")
        baseline_options.add_argument('--cummulative', action='store_true',
                                      help="train incrementally on data of all contexts so far")

    param_reg = parser.add_argument_group('Parameter Regularization')
    param_reg.add_argument('--ewc', action='store_true',
                           help="select defaults for 'EWC' (Kirkpatrick et al, 2017)")
    param_reg.add_argument('--si', action='store_true', help="select defaults for 'SI' (Zenke et al, 2017)")
    param_reg.add_argument('--epsilon', type=float, default=0.1, dest="epsilon",
                           help="-> SI: dampening parameter")

    ## Functional regularization
    func_reg = parser.add_argument_group('Functional Regularization')
    func_reg.add_argument('--lwf', action='store_true', help="select defaults for 'LwF' (Li & Hoiem, 2017)")

    ## Memory buffer parameters (if data is stored)
    buffer = parser.add_argument_group('Memory Buffer Parameters')
    if not compare_replay:
        buffer.add_argument('--budget', type=float, help="how many samples can be stored{}".format(
            " (total budget)" if no_boundaries else " of each class?"
        ), default=1000 if no_boundaries else None)
    if not no_boundaries:
        buffer.add_argument('--use-full-capacity', action='store_true',
                            help="use budget of future classes to initially store more")
    if main and not no_boundaries:
        buffer.add_argument('--sample-selection', type=str, choices=['random', 'herding', 'fromp'])
        buffer.add_argument('--add-buffer', action='store_true',
                            help="add memory buffer to current context's training data")

    ## Replay
    replay_params = parser.add_argument_group('Replay')
    if main:
        replay_params.add_argument('--er', action='store_true')
        replay_params.add_argument('--dgr', action='store_true')
        replay_choices = ['none', 'current', 'buffer'] if no_boundaries else ['none', 'all', 'generative',
                                                                              'current', 'buffer']
        replay_params.add_argument('--replay', type=str, default='none', choices=replay_choices)
        replay_params.add_argument('--use-replay', type=str, default='normal', choices=['normal', 'inequality', 'both'])
        replay_params.add_argument('--z-dim', type=int, help='size of latent representation (if used, def=100)')
        replay_params.add_argument('--gen-epoch', type=int)
        replay_params.add_argument('--disc-epoch', type=int)
        # ---> Explanation for these three ways to use replay:
        # - "normal":      add the loss on the replayed data to the loss on the data of the current context
        # - "inequality":  use the gradient of the loss on the replayed data as an inequality constraint (as in A-GEM)
        # - "both":        do both of the above
        replay_params.add_argument('--agem', action='store_true',
                                   help="select defaults for 'A-GEM' (Chaudhry et al, 2019)")
        replay_params.add_argument('--bir', action='store_true',
                               help="select defaults for 'BI-R' (van de Ven et al, 2020)")
    replay_params.add_argument('--eps-agem', type=float, default=1e-7,
                               help="parameter to ensure numerical stability of A-GEM")

    if (not compare_replay) and (not no_boundaries):
        if main:
            replay_params.add_argument('--brain-inspired', action='store_true',
                                       help="select defaults for 'bir' (van de Ven et al, 2020)")
            replay_params.add_argument('--feedback', action="store_true",
                                       help="equip main model with feedback connections")
            replay_params.add_argument('--prior', type=str, default="standard", choices=["standard", "GMM"])
            replay_params.add_argument('--per-class', action='store_true',
                                       help="if selected, each class has its own modes")
    ## Template-based classification
    if not compare_replay:
        templ_cl = parser.add_argument_group('Template-Based Classification')
        if main:
            templ_cl.add_argument('--icarl', action='store_true',
                                  help="select defaults for '{}iCaRL' (Rebuffi et al, 2017)".format(
                                      'Modified ' if no_boundaries else ''
                                  ))
            templ_cl.add_argument('--multi-classifier', action='store_true')
            templ_cl.add_argument('--gen-classifier', action='store_true',
                                  help="use 'Generative Classifier' (van de Ven et al, 2021)")

    return parser
