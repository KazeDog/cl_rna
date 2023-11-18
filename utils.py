import os
import numpy as np
import pickle
import copy
import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,TensorDataset
from torchvision import transforms
import layers




##-------------------------------------------------------------------------------------------------------------------##

######################
## Random utilities ##
######################

def checkattr(args, attr):
    '''Check whether attribute exists, whether it's a boolean and whether its value is True.'''
    return hasattr(args, attr) and type(getattr(args, attr))==bool and getattr(args, attr)


def to_onehot(seq):
    """
    Inputs:
        seq: RNA seqs in string capitalized form
    """
    out = F.one_hot(seq)
    out_size = list(seq.size())
    length = len(out_size)
    if length == 1:
        out = out[:, 1:]
    elif length == 2:
        out = out[:, :, 1:]
    elif length == 3:
        out = out[:, :, :, 1:]
    return out

##-------------------------------------------------------------------------------------------------------------------##

#############################
## Data-handling functions ##
#############################

def get_data_loader(dataset, batch_size, cuda=False, drop_last=False):
    '''Return <DataLoader>-object for the provided <DataSet>-object [dataset].'''

    # If requested, make copy of original dataset to add augmenting transform (without altering original dataset)
    dataset_ = dataset

    # Create and return the <DataLoader>-object
    return DataLoader(
        dataset_, batch_size=batch_size, shuffle=True, drop_last=drop_last,
        **({'num_workers': 0, 'pin_memory': True} if cuda else {})
    )

def to_one_hot(y, classes):
    '''Convert a nd-array with integers [y] to a 2D "one-hot" tensor.'''
    c = np.zeros(shape=[len(y), classes], dtype='float32')
    c[range(len(y)), y] = 1.
    c = torch.from_numpy(c)
    return c

##-------------------------------------------------------------------------------------------------------------------##

##########################################
## Object-saving and -loading functions ##
##########################################

def save_object(object, path):
    with open(path + '.pkl', 'wb') as f:
        pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)

def load_object(path):
    with open(path + '.pkl', 'rb') as f:
        return pickle.load(f)

##-------------------------------------------------------------------------------------------------------------------##

#########################################
## Model-saving and -loading functions ##
#########################################

def save_checkpoint(model, model_dir, verbose=True, name=None):
    '''Save state of [model] as dictionary to [model_dir] (if name is None, use "model.name").'''
    # -name/path to store the checkpoint
    name = model.name if name is None else name
    path = os.path.join(model_dir, name)
    # -if required, create directory in which to save checkpoint
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # -create the dictionary containing the checkpoint
    checkpoint = {'state': model.state_dict()}
    if hasattr(model, 'mask_dict') and model.mask_dict is not None:
        checkpoint['mask_dict'] = model.mask_dict
    # -(try to) save the checkpoint
    try:
        torch.save(checkpoint, path)
        if verbose:
            print(' --> saved model {name} to {path}'.format(name=name, path=model_dir))
    except OSError:
        print(" --> saving model '{}' failed!!".format(name))

def load_checkpoint(model, model_dir, verbose=True, name=None, strict=True):
    '''Load saved state (in form of dictionary) at [model_dir] (if name is None, use "model.name") to [model].'''
    # -path from where to load checkpoint
    name = model.name if name is None else name
    path = os.path.join(model_dir, name)
    # load parameters (i.e., [model] will now have the state of the loaded model)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state'], strict=strict)
    if 'mask_dict' in checkpoint:
        model.mask_dict = checkpoint['mask_dict']
    # notify that we succesfully loaded the checkpoint
    if verbose:
        print(' --> loaded checkpoint of {name} from {path}'.format(name=name, path=model_dir))

##-------------------------------------------------------------------------------------------------------------------##


########################################
## Parameter-initialization functions ##
########################################

def weight_reset(m):
    '''Reinitializes parameters of [m] according to default initialization scheme.'''
    if isinstance(m, nn.Transformer) or isinstance(m, nn.Linear) or isinstance(m, layers.LinearExcitability):
        m.reset_parameters()

def weight_init(model, strategy="xavier_normal", std=0.01):
    '''Initialize weight-parameters of [model] according to [strategy].

    [xavier_normal]     "normalized initialization" (Glorot & Bengio, 2010) with Gaussian distribution
    [xavier_uniform]    "normalized initialization" (Glorot & Bengio, 2010) with uniform distribution
    [normal]            initialize with Gaussian(mean=0, std=[std])
    [...]               ...'''

    # If [model] has an "list_init_layers"-attribute, only initialize parameters in those layers
    if hasattr(model, "list_init_layers"):
        module_list = model.list_init_layers()
        parameters = [p for m in module_list for p in m.parameters()]
    else:
        parameters = [p for p in model.parameters()]

    # Initialize all weight-parameters (i.e., with dim of at least 2)
    for p in parameters:
        if p.dim() >= 2:
            if strategy=="xavier_normal":
                nn.init.xavier_normal_(p)
            elif strategy=="xavier_uniform":
                nn.init.xavier_uniform_(p)
            elif strategy=="normal":
                nn.init.normal_(p, std=std)
            else:
                raise ValueError("Invalid weight-initialization strategy {}".format(strategy))

def bias_init(model, strategy="constant", value=0.01):
    '''Initialize bias-parameters of [model] according to [strategy].

    [zero]      set them all to zero
    [constant]  set them all to [value]
    [positive]  initialize with Uniform(a=0, b=[value])
    [any]       initialize with Uniform(a=-[value], b=[value])
    [...]       ...'''

    # If [model] has an "list_init_layers"-attribute, only initialize parameters in those layers
    if hasattr(model, "list_init_layers"):
        module_list = model.list_init_layers()
        parameters = [p for m in module_list for p in m.parameters()]
    else:
        parameters = [p for p in model.parameters()]

    # Initialize all weight-parameters (i.e., with dim of at least 2)
    for p in parameters:
        if p.dim() == 1:
            ## NOTE: be careful if excitability-parameters are added to the model!!!!
            if strategy == "zero":
                nn.init.constant_(p, val=0)
            elif strategy == "constant":
                nn.init.constant_(p, val=value)
            elif strategy == "positive":
                nn.init.uniform_(p, a=0, b=value)
            elif strategy == "any":
                nn.init.uniform_(p, a=-value, b=value)
            else:
                raise ValueError("Invalid bias-initialization strategy {}".format(strategy))