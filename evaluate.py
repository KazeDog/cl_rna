import math

import numpy as np
import torch
from torch.nn import functional as F
from sklearn import metrics
from utils import get_data_loader, checkattr


####-----------------------------####
####----CLASSIFIER EVALUATION----####
####-----------------------------####

def test(model, dataset, pos_label, batch_size=128, test_size=None, verbose=True, context_id=None, allowed_classes=None,
         no_context_mask=False, method=None, tofile=False, args=None, **kwargs):
    '''Evaluate accuracy (= proportion of samples classified correctly) of a classifier ([model]) on [dataset].

    [allowed_classes]   None or <list> containing all "active classes" between which should be chosen
                            (these "active classes" are assumed to be contiguous)'''

    # Get device-type / using cuda?
    device = model.device if hasattr(model, 'device') else model._device()
    cuda = model.cuda if hasattr(model, 'cuda') else model._is_on_cuda()

    # Set model to eval()-mode
    mode = model.training
    model.eval()
    # Apply context-specifc "gating-mask" for each hidden fully connected layer (or remove it!)
    if hasattr(model, "mask_dict") and model.mask_dict is not None:
        if no_context_mask:
            model.reset_XdGmask()
        else:
            model.apply_XdGmask(context=context_id + 1)

    label_correction = 0 if checkattr(model, 'stream_classifier') or (allowed_classes is None) else allowed_classes[1]
    if label_correction != 0:
        pos_label = 1
    if model.label == "SeparateClassifiers":
        model = getattr(model, 'context{}'.format(context_id + 1))
        allowed_classes = None

    # Loop over batches in [dataset]
    data_loader = get_data_loader(dataset, batch_size, cuda=cuda)
    total_tested = total_correct = 0
    vlosses, vlabels_total, vpredicted_total = [], [], []
    for x, y in data_loader:
        # -break on [test_size] (if "None", full dataset is used)
        if test_size:
            if total_tested >= test_size:
                break
        # -if the model is a "stream-classifier", add context
        if checkattr(model, 'stream_classifier'):
            context_tensor = torch.tensor([context_id] * x.shape[0]).to(device)
        # -evaluate model (if requested, only on [allowed_classes])
        with torch.no_grad():
            if checkattr(model, 'stream_classifier'):
                scores = model.classify(x.to(device), context=context_tensor)
            else:
                scores = model.classify(x.to(device), allowed_classes=allowed_classes)
        vscore = torch.argmax(scores.cpu(), 1)
        _, predicted = torch.max(scores.cpu(), 1)
        if checkattr(args, 'icarl'):
            predicted = predicted.cpu().numpy().tolist()
            predicted = [0 if p % 2 == 0 else math.ceil(p / 2) for p in predicted]
            predicted = torch.LongTensor(predicted)

        if label_correction != 0:
            y = y.cpu().numpy().tolist()
            y = [label - label_correction + 1 if label != 0 else label for label in y]
            y = torch.LongTensor(y)
        loss = F.cross_entropy(input=scores.cpu(), target=y, reduction='mean')
        vlosses.append(loss)
        vlabels_total += y
        vpredicted_total += predicted
    accuracy = metrics.accuracy_score(vlabels_total, vpredicted_total)
    precision = metrics.precision_score(vlabels_total, vpredicted_total, average='weighted')
    recall = metrics.recall_score(vlabels_total, vpredicted_total, average='weighted')
    f1 = metrics.f1_score(vlabels_total, vpredicted_total, average='weighted')

    fpr, tpr, threshold = metrics.roc_curve(vlabels_total, vpredicted_total, pos_label=pos_label)
    auc = metrics.auc(fpr, tpr)
    model.train(mode=mode)
    if verbose:
        print('test context_id: {}, accuracy: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}, AUC: {:.4f}'
              .format(context_id + 1, accuracy, precision, recall, f1, auc))
    if tofile and method is not None:
        writefile = '{}/{}_{}.csv'.format(args.r_dir, method, args.index)
        with open(writefile, 'a') as wf:
            wf.write('test context_id: {},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}'
              .format(context_id + 1, accuracy, precision, recall, f1, auc))
            wf.write('\n')
    return np.mean(vlosses), accuracy, precision, recall, f1, auc


def test_all_so_far(model, datasets, current_context, test_size=None, no_context_mask=False,
                    visdom=None, summary_graph=True, plotting_dict=None, verbose=True, method=None, tofile=False, args=None):
    '''Evaluate accuracy of a classifier (=[model]) on all contexts so far (= up to [current_context]) using [datasets].

    [visdom]      None or <dict> with name of "graph" and "env" (if None, no visdom-plots are made)'''

    n_contexts = len(datasets)

    # Evaluate accuracy of model predictions
    # - in the academic CL setting:  for all contexts so far, reporting "0" for future contexts
    # - in task-free stream setting (current_context==None): always for all contexts
    vlosses, accuracys, precisions, recalls, f1s, aucs = [], [], [], [], [], []
    for i in range(n_contexts):
        if (current_context is None) or (i + 1 <= current_context):
            allowed_classes = None
            if model.scenario == 'task' and not checkattr(model, 'singlehead'):
                # allowed_classes = list(range(model.classes_per_context * i, model.classes_per_context * (i + 1)))
                if i == 0:
                    allowed_classes = [0, 1]
                else:
                    allowed_classes = [list(
                        range(index + 1)
                    )[::index] for index in range(i + 2) if index > 0][-1]
            vloss, accuracy, precision, recall, f1, auc = test(model, datasets[i], test_size=test_size,
                                                               verbose=verbose,
                                                               allowed_classes=allowed_classes,
                                                               no_context_mask=no_context_mask, context_id=i,
                                                               pos_label= i + 1, method=method, tofile=tofile,
                                                               args=args)
            vlosses.append(vloss)
            accuracys.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            aucs.append(auc)
        else:
            vlosses.append(0)
            accuracys.append(0)
            precisions.append(0)
            recalls.append(0)
            f1s.append(0)
            aucs.append(0)
    if current_context is None:
        current_context = i + 1
    average_vlosses = sum([vlosses[context_id] for context_id in range(current_context)]) / current_context
    average_accuracys = sum([accuracys[context_id] for context_id in range(current_context)]) / current_context
    average_precisions = sum([precisions[context_id] for context_id in range(current_context)]) / current_context
    average_recalls = sum([recalls[context_id] for context_id in range(current_context)]) / current_context
    average_f1s = sum([f1s[context_id] for context_id in range(current_context)]) / current_context
    average_aucs = sum([aucs[context_id] for context_id in range(current_context)]) / current_context

    # Print results on screen
    if verbose:
        print(' => ave accuracy: {:.4f}, ave precision: {:.4f}, ave recall: {:.4f}, ave f1: {:.4f}, ave auc: {:.4f}'
              .format(average_accuracys, average_precisions, average_recalls, average_f1s, average_aucs))

    if tofile and method is not None:
        writefile = '{}/{}_{}.csv'.format(args.r_dir, method, args.index)
        with open(writefile, 'a') as wf:
            wf.write('Context: {} => average,{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}'
              .format(current_context, average_accuracys, average_precisions, average_recalls, average_f1s, average_aucs))
            wf.write('\n')

    return average_vlosses, average_accuracys, average_precisions, average_recalls, average_f1s, average_aucs
