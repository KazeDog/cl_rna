import os
import random

import torch
import argparse
import time
from torch import optim
import numpy as np


import options
import utils
from utils import checkattr
from data.load import get_context_set
from param_values import set_method_options, set_default_values
import define_models
import define_train
from evaluate import test

## Function for specifying input-options and organizing / checking them
def handle_inputs():
    # Set indicator-dictionary for correctly retrieving / checking input options
    kwargs = {'main': True}
    # Define input options
    parser = options.define_args(filename="main", description='Run an individual continual learning experiment '
                                                              'using the "academic continual learning setting".')
    parser = options.add_general_options(parser, **kwargs)
    parser = options.add_eval_options(parser, **kwargs)
    parser = options.add_problem_options(parser, **kwargs)
    parser = options.add_model_options(parser, **kwargs)
    parser = options.add_train_options(parser, **kwargs)
    parser = options.add_cl_options(parser, **kwargs)
    # Parse, process and check chosen options
    # parser = argparse.ArgumentParser("")
    args = parser.parse_args()
    set_method_options(args)                         # -if a method's "convenience"-option is chosen, select components
    set_default_values(args, also_hyper_params=True) # -set defaults, some are based on chosen scenario / experiment
    # check_for_errors(args, **kwargs)                 # -check whether incompatible options are selected
    return args


def run(args, verbose=False):

    # Create plots- and results-directories if needed
    if not os.path.isdir(args.r_dir):
        os.mkdir(args.r_dir)
    if checkattr(args, 'pdf') and not os.path.isdir(args.p_dir):
        os.mkdir(args.p_dir)

    # If only want param-stamp, get it printed to screen and exit
    # if checkattr(args, 'get_stamp'):
    #     print(get_param_stamp_from_args(args=args))
    #     exit()

    # Use cuda?
    cuda = torch.cuda.is_available() and args.cuda
    device = torch.device("cuda" if cuda else "cpu")

    # Report whether cuda is used
    if verbose:
        print("CUDA is {}used".format("" if cuda else "NOT(!!) "))

    # Set random seeds

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    #-------------------------------------------------------------------------------------------------#

    #----------------#
    #----- DATA -----#
    #----------------#

    # Prepare data for chosen experiment
    if verbose:
        print("\n\n " +' LOAD DATA '.center(70, '*'))
    (train_datasets, valid_datasets, test_datasets), config = get_context_set(
            scenario=args.scenario, contexts=args.contexts, classes=args.classes,
        seqlen=args.seqlen, data_dir=args.data_dir
    )

    #-------------------------------------------------------------------------------------------------#


    #----------------------#
    #----- CLASSIFIER -----#
    #----------------------#

    # Define the classifier
    if verbose:
        print("\n\n " + ' DEFINE THE CLASSIFIER '.center(70, '*'))
    model = define_models.define_classifier(args=args, config=config, device=device, use_embedding=True)

    # Some type of classifiers consist of multiple networks
    print("len(train_datasets)=",len(train_datasets))

    n_networks = 1

    print("n_networks=", n_networks)

    for network_id in range(n_networks):
        model_to_set = model
        # ... initialize / use pre-trained / freeze model-parameters, and
        define_models.init_params(model_to_set, args)
        # ... define optimizer (only include parameters that "requires_grad")
        if not checkattr(args, 'fromp'):
            model_to_set.optim_list = [{'params': filter(lambda p: p.requires_grad, model_to_set.parameters()),
                                        'lr': args.lr}]
            model_to_set.optim_type = args.optimizer
            if model_to_set.optim_type in ("adam", "adam_reset"):
                model_to_set.optimizer = optim.Adam(model_to_set.optim_list, betas=(0.9, 0.999))
            elif model_to_set.optim_type=="sgd":
                model_to_set.optimizer = optim.SGD(model_to_set.optim_list,
                                                   momentum=args.momentum if hasattr(args, 'momentum') else 0.)

    # On what scenario will model be trained? If needed, indicate whether singlehead output / how to set active classes.
    model.scenario = args.scenario
    model.classes_per_context = config['classes_per_context']
    model.singlehead = checkattr(args, 'singlehead')
    model.neg_samples = args.neg_samples if hasattr(args, 'neg_samples') else "all-so-far"

    if args.save:
        writefile = '{}/{}_{}.csv'.format(args.r_dir, args.method, args.index)
        with open(writefile, 'a') as wf:
            wf.write('name,accuracy,precision,recall,f1,AUC\n')

    #-------------------------------------------------------------------------------------------------#


    #--------------------#
    #----- TRAINING -----#
    #--------------------#

    # Should a baseline be used (i.e., 'joint training' or 'cummulative training')?
    baseline = 'joint' if checkattr(args, 'joint') else ('cummulative' if checkattr(args, 'cummulative') else 'none')

    # Train model
    if args.train:
        if verbose:
            print('\n\n' + ' TRAINING '.center(70, '*'))
        # -keep track of training-time
        if args.time:
            start = time.time()
        # -select correct training function
        train_fn = define_train.define_trainway(args, model, train_datasets, valid_datasets, baseline)
        # print('train_fn=', train_fn)

    #-------------------------------------------------------------------------------------------------#

    #----------------------#
    #----- EVALUATION -----#
    #----------------------#

    if verbose:
        print('\n\n' + ' EVALUATION '.center(70, '*'))

    # Evaluate accuracy of final model on full test-set
    if verbose:
        print("\n Accuracy of final model on test-set:")

    accuracys, precisions, recalls, f1s, aucs = [], [], [], [], []
    for i in range(args.contexts):
        _, accuracy, precision, recall, f1, auc = test(
            model, test_datasets[i], pos_label=i + 1,
            verbose=False, test_size=None, context_id=i, allowed_classes=list(
                range(config['classes_per_context'] * i, config['classes_per_context'] * (i + 1))
            ) if (args.scenario == "task" and not checkattr(args, 'singlehead')) else None,
            args=args
        )
        if verbose:
            print('- Context {}: accuracy: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}, AUC: {:.4f}'
                  .format(i + 1, accuracy, precision, recall, f1, auc))
        if args.save:
            with open(writefile, 'a') as wf:
                wf.write('-Context {},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}'
                         .format(i + 1, accuracy, precision,
                                 recall, f1, auc))
                wf.write('\n')
        accuracys.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        aucs.append(auc)
    average_accuracys = sum(accuracys) / args.contexts
    average_precisions = sum(precisions) / args.contexts
    average_recalls = sum(recalls) / args.contexts
    average_f1s = sum(f1s) / args.contexts
    average_aucs = sum(aucs) / args.contexts
    if verbose:
        print('=> over all {} contexts: ave accuracy: {:.4f}, ave precision: {:.4f}, ave recall: {:.4f}, '
              'ave f1: {:.4f}, ave auc: {:.4f}\n\n'.format(args.contexts, average_accuracys, average_precisions,
                                                           average_recalls, average_f1s, average_aucs))
    if args.save:
        with open(writefile, 'a') as wf:
            wf.write('=> over all {} contexts,{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}'
                     .format(args.contexts, average_accuracys, average_precisions,
                                                               average_recalls, average_f1s, average_aucs))
            wf.write('\n')




if __name__ == '__main__':
    # -load input-arguments
    args = handle_inputs()
    # -run experiment

    arg_vars = vars(args)
    print("# ===============================================")
    print("## parameters: ")
    for arg_key in arg_vars.keys():
        if arg_key != 'func':
            print("{}:\n\t{}".format(arg_key, arg_vars[arg_key]))
    print("# ===============================================")
    run(args, verbose=True)