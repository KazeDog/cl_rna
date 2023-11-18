import os

import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset

from data.load import get_test_set
import math
from sklearn import metrics
import numpy as np

from utils import get_data_loader, checkattr
import options
from utils import checkattr
from param_values import set_method_options, set_default_values
import define_models

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

def load_args():
    args = handle_inputs()
    arg_vars = vars(args)
    print("# ===============================================")
    print("## parameters: ")
    for arg_key in arg_vars.keys():
        if arg_key != 'func':
            print("{}:\n\t{}".format(arg_key, arg_vars[arg_key]))
    print("# ===============================================")

    return args

def load_testdata(args):
    print("\n\n " + ' LOAD DATA '.center(70, '*'))
    valid_datasets, config = get_test_set(
        scenario=args.scenario, contexts=args.contexts, classes=args.classes,
        seqlen=args.seqlen, data_dir=args.data_dir,
        singlehead=checkattr(args, 'singlehead'), train_set_per_class=checkattr(args, 'gen_classifier')
    )
    return valid_datasets, config

def define_classifier(args, config, device, model_type):
    global model
    if model_type == 'si':
        model = define_models.define_si_classifier(args=args, config=config, device=device)
    elif model_type == 'lwf':
        model = define_models.define_lwf_classifier(args=args, config=config, device=device)
    elif model_type == 'er':
        model = define_models.define_er_classifier(args=args, config=config, device=device)
    elif model_type == 'dgr':
        model = define_models.define_dgr_classifier(args=args, config=config, device=device)
    elif model_type == 'icarl':
        model = define_models.define_icarl_classifier(args=args, config=config, device=device)
    else:
        model = define_models.define_standard_classifier(args=args, config=config, device=device)
    return model

def load_model(args, valid_datasets, config, method):


    cuda = torch.cuda.is_available() and args.cuda
    device = torch.device("cuda" if cuda else "cpu")


    model = define_classifier(args=args, config=config, device=device, model_type=method)
    print("len(valid_datasets)=", len(valid_datasets))

    n_networks = len(valid_datasets) if checkattr(args, 'separate_networks') else (
        config['output_units'] if checkattr(args, 'gen_classifier') else 1)

    print("n_networks=", n_networks)

    for network_id in range(n_networks):
        model_to_set = getattr(model, 'context{}'.format(network_id + 1)) if checkattr(args, 'separate_networks') else (
            getattr(model, 'vae{}'.format(network_id)) if checkattr(args, 'gen_classifier') else model
        )
        # ... initialize / use pre-trained / freeze model-parameters, and
        define_models.init_params(model_to_set, args)
        # ... define optimizer (only include parameters that "requires_grad")
        if not checkattr(args, 'fromp'):
            model_to_set.optim_list = [{'params': filter(lambda p: p.requires_grad, model_to_set.parameters()),
                                        'lr': args.lr}]
            model_to_set.optim_type = args.optimizer
            if model_to_set.optim_type in ("adam", "adam_reset"):
                model_to_set.optimizer = optim.Adam(model_to_set.optim_list, betas=(0.9, 0.999))
            elif model_to_set.optim_type == "sgd":
                model_to_set.optimizer = optim.SGD(model_to_set.optim_list,
                                                   momentum=args.momentum if hasattr(args, 'momentum') else 0.)

    # On what scenario will model be trained? If needed, indicate whether singlehead output / how to set active classes.
    model.scenario = args.scenario
    model.classes_per_context = config['classes_per_context']
    model.singlehead = checkattr(args, 'singlehead')
    model.neg_samples = args.neg_samples if hasattr(args, 'neg_samples') else "all-so-far"

    return model

def test_all(model, test_datasets, current_context, method=None):

    device = model.device if hasattr(model, 'device') else model._device()
    cuda = model.cuda if hasattr(model, 'cuda') else model._is_on_cuda()

    # Set model to eval()-mode
    mode = model.training
    model.eval()
    # print('test_datasets length=', len(test_datasets))
    test_datasets = ConcatDataset(test_datasets[:current_context])
    # print('len test_datasets=', len(test_datasets))
    data_loader = get_data_loader(test_datasets, 128, cuda=cuda)
    vlabels_total, vpredicted_total, scores_total = [], [], []
    for x, y in data_loader:
        with torch.no_grad():
            scores = model.classify(x.to(device))

        vscore, predicted = torch.max(scores.cpu(), 1)
        softmax = nn.Softmax(1)
        scores = softmax(scores)
        scores = scores.cpu()
        if method == 'icarl':
            predicted = predicted.cpu().numpy().tolist()
            predicted = [0 if p % 2 == 0 else math.ceil(p / 2) for p in predicted]
            predicted = torch.LongTensor(predicted)
        vlabels_total += y
        vpredicted_total += predicted
        scores = np.array(scores)
        scores_total.append(scores)

    scores_total = np.concatenate(scores_total)

    accuracy = metrics.accuracy_score(vlabels_total, vpredicted_total)
    precision = metrics.precision_score(vlabels_total, vpredicted_total, average='weighted')
    recall = metrics.recall_score(vlabels_total, vpredicted_total, average='weighted')
    f1 = metrics.f1_score(vlabels_total, vpredicted_total, average='weighted')
    if method == 'icarl':
        auc = 0
    else:
        auc = metrics.roc_auc_score(vlabels_total, scores_total, average='weighted', multi_class='ovr')
    # auc = metrics.roc_auc_score(vlabels_total, scores_total, average='weighted', multi_class='ovr')
    print(' => over all 10 contexts: accuracy: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}, AUC: {:.4f}'
          .format(accuracy, precision, recall, f1, auc))
    with open(writefile, 'a') as wf:
        wf.write('=> over all 10 contexts,{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}'
                 .format(accuracy, precision, recall, f1, auc))
        wf.write('\n')


if __name__ == '__main__':
    # path = '/mnt/8t/qjb/cl-rna/results/original'
    path = '/'

    if not os.path.isdir(f'{path}/test'):
        os.mkdir(f'{path}/test')

    args = load_args()
    test_datasets, config = load_testdata(args)
    methods = ['none', 'joint', 'si', 'lwf', 'er', 'dgr', 'icarl']
    # methods = ['dgr']
    for method in methods:
        model = load_model(args, test_datasets, config, method)
        writefile = '{}/test_{}.csv'.format(f'{path}/test', method)
        with open(writefile, 'a') as wf:
            wf.write('name,accuracy,precision,recall,f1,AUC\n')
        for i in range(5):

            if method in ['none', 'joint']:
                model.load_state_dict(
                    torch.load(f'/baseline/{method}_classifier_{i+1}_context10.pt'))
            elif method in ['dgr']:
                model.load_state_dict(
                    torch.load(f'/dgr/{i+1}_classifier_{method}_context10.pt'))
            elif method in ['icarl']:

                model = torch.load(f'/{method}/{i+1}_classifier_{method}_context10.pt')
            else:
                model.load_state_dict(
                    torch.load(f'/{method}/{i+1}_classifier_{method}_context10.pt'),
                    strict=False)
            test_all(model, test_datasets, 10, method)
