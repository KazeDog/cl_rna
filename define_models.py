import utils
from utils import checkattr
from torch import optim, nn


def define_classifier(args, config, device, use_embedding, depth=0, stream=False):
    global model

    if checkattr(args, 'si'):
        model = define_si_classifier(args=args, config=config, device=device)
    elif checkattr(args, 'lwf'):
        model = define_lwf_classifier(args=args, config=config, device=device)
    elif checkattr(args, 'er'):
        model = define_er_classifier(args=args, config=config, device=device)
    elif checkattr(args, 'dgr'):
        model = define_dgr_classifier(args=args, config=config, device=device)
    elif checkattr(args, 'icarl'):
        model = define_icarl_classifier(args=args, config=config, device=device)
    else:
        model = define_standard_classifier(args=args, config=config, device=device)
    return model



## Function for (re-)initializing the parameters of [model]
def init_params(model, args, verbose=False):

    ## Initialization
    # - reinitialize all parameters according to default initialization
    model.apply(utils.weight_reset)
    # - initialize parameters according to chosen custom initialization (if requested)
    if hasattr(args, 'init_weight') and not args.init_weight=="standard":
        utils.weight_init(model, strategy="xavier_normal")
    if hasattr(args, 'init_bias') and not args.init_bias=="standard":
        utils.bias_init(model, strategy="constant", value=0.01)

def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

##-------------------------------------------------------------------------------------------------------------------##

def define_si_classifier(args, config, device):
    # Import required model
    # Specify model
    from methods.parameter_regularization.si.model import Classifier
    print('now define_standard_classifier')
    model = Classifier(
        classes=config['output_units'],
        seqlen=args.seqlen,

        excit_buffer=True,
        phantom=checkattr(args, 'fisher_kfac'),
        epsilon=args.epsilon if hasattr(args, 'epsilon') else 0.1
    ).to(device)

    return model

def define_lwf_classifier(args, config, device):
    # Import required model
    # Specify model
    from methods.functional_regularization.lwf.model import Classifier
    print('now define_standard_classifier')
    model = Classifier(
        classes=config['output_units'],
        seqlen= args.seqlen,
    ).to(device)
    # Return model
    return model

def define_er_classifier(args, config, device):
    # Import required model
    # Specify model
    from methods.repaly.er.model import Classifier
    print('now define_standard_classifier')
    model = Classifier(
        classes=config['output_units'],
        seqlen= args.seqlen,
        excit_buffer=True,
        phantom=checkattr(args, 'fisher_kfac'),
        epsilon=args.epsilon if hasattr(args, 'epsilon') else 0.1
    ).to(device)
    if hasattr(args, 'replay'):
        model.replay_mode = args.replay
    model.use_memory_buffer = True
    model.budget = args.budget
    model.use_full_capacity = checkattr(args, 'use_full_capacity')
    model.sample_selection = args.sample_selection if hasattr(args, 'sample_selection') else 'random'
    model.norm_exemplars = (model.sample_selection == "herding")
    model.replay_targets = "soft" if checkattr(args, 'distill') else "hard"
    model.binaryCE = checkattr(args, 'bce')
    model.binaryCE_distill = checkattr(args, 'bce_distill')
    return model

def define_dgr_classifier(args, config, device):
    # Import required model
    # Specify model
    from methods.repaly.dgr.classifier import Classifier
    print('now define_standard_classifier')
    model = Classifier(
        classes=config['output_units'],
        seqlen= args.seqlen,
        excit_buffer=True,
        phantom=checkattr(args, 'fisher_kfac'),
        epsilon=args.epsilon if hasattr(args, 'epsilon') else 0.1
    ).to(device)
    # Return model
    if hasattr(args, 'replay'):
        model.replay_mode = args.replay
    model.replay_targets = "soft" if checkattr(args, 'distill') else "hard"
    model.binaryCE = checkattr(args, 'bce')
    model.binaryCE_distill = checkattr(args, 'bce_distill')

    return model
def define_icarl_classifier(args, config, device):
    # Import required model
    # Specify model
    from methods.template_based_classification.icarl.model import Classifier
    print('now define_standard_classifier')
    model = Classifier(
        classes=config['output_units'],
        seqlen= args.seqlen,
        excit_buffer=True,
        phantom=checkattr(args, 'fisher_kfac'),
        epsilon=args.epsilon if hasattr(args, 'epsilon') else 0.1
    ).to(device)
    # Return model
    if hasattr(args, 'replay'):
        model.replay_mode = args.replay
    model.use_memory_buffer = True
    model.budget = args.budget
    model.use_full_capacity = checkattr(args, 'use_full_capacity')
    model.norm_exemplars = (model.sample_selection == "herding")
    model.replay_targets = "soft" if checkattr(args, 'distill') else "hard"
    print('model.replay_targets=', model.replay_targets)
    return model


def define_standard_classifier(args, config, device, depth=1):
    # Import required model
    from methods.baseline.model import Classifier
    # Specify model
    print('now define_standard_classifier')
    model = Classifier(
        classes=config['output_units'],
        seqlen=args.seqlen,
        fc_layers=3,
        excit_buffer=True,
        phantom=checkattr(args, 'fisher_kfac')
    ).to(device)
    return model

