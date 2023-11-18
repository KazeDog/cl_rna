import copy

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
import time

from torch.utils.data import ConcatDataset
from evaluate import test_all_so_far
from save_model import save_classifier
from utils import get_data_loader, checkattr
import model_utils.loss_functions as lf


#------------------------------------------------------------------------------------------------------------#

def train_cl(args, model, train_datasets, valid_datasets, min_epoch_num, max_epoch_num, batch_size=32,
             step_interval=10, baseline='none', **kwargs):

    # Set model in training-mode
    model.train()

    # Use cuda?
    cuda = model._is_on_cuda()
    device = model._device()


    # Are there different active classes per context (or just potentially a different mask per context)?
    per_context = False
    per_context_singlehead = False

    start = time.time()
    # Loop over all contexts.
    for context, train_dataset in enumerate(train_datasets, 1):
        print('context=', context)
        print('train_dataset=', train_dataset)
        print('len train_dataset=', len(train_dataset))

        # If using the "joint" baseline, skip to last context, as model is only be trained once on data of all contexts
        if baseline=='joint':
            if context<len(train_datasets):
                continue
            else:
                baseline = "cummulative"

        # If using the "cummulative" (or "joint") baseline, create a large training dataset of all contexts so far
        if baseline=="cummulative" and (not per_context):
            train_dataset = ConcatDataset(train_datasets[:context])
        # -but if "cummulative"+[per_context]: training on each context must be separate, as a trick to achieve this,
        #                                      all contexts so far are treated as replay (& there is no current batch)

        print('baseline=', baseline)

        training_dataset = train_dataset

        active_classes = list(range(context * int(model.classes_per_context / 2) + 1))
        print('active_classes=', active_classes)

        if model.optim_type=="adam_reset":
            model.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))

        print('per_context=', per_context)
        if per_context:
            up_to_context = context if baseline=="cummulative" else context-1
            print('up_to_context=', up_to_context)
            iters_left_previous = [1]*up_to_context
            print('iters_left_previous=', iters_left_previous)
            data_loader_previous = [None]*up_to_context
            print('data_loader_previous=', data_loader_previous)

        data_loader = get_data_loader(training_dataset, batch_size, cuda=cuda, drop_last=True)
        total_step = len(data_loader)
        print("total_step: {}".format(total_step))
        curr_best_accuracy = 0
        iters = max_epoch_num * len(data_loader)
        for epoch in range(max_epoch_num):
            curr_best_accuracy_epoch = 0

            # -----------------Collect data------------------#
            for i, sfeatures in enumerate(data_loader):
                #####-----CURRENT BATCH-----#####
                if baseline=="cummulative" and per_context:
                    x = y = scores = None
                else:
                    x, y = sfeatures                             #--> sample training data of current context
                    # print('y1=', y)
                    if per_context and not per_context_singlehead:
                        y = y.cpu().numpy().tolist()
                        y = [label - context + 1 if label != 0 else label for label in y]
                        y = torch.LongTensor(y)
                    x, y = x.to(device), y.to(device)                    #--> transfer them to correct device

                    scores = None

                tlosses, accuracy = train_a_batch(model, x, y, scores=scores,
                                                  rnt=1. / context,
                                                  active_classes=active_classes,
                                                  context=context)

                if (((i + 1) % step_interval == 0 or i == total_step - 1) and baseline == 'cummulative') or (
                        i == total_step - 1 and baseline == 'none'):
                    v_loss, v_acc, v_prec, v_recall, v_f1, auc= test_all_so_far(model, valid_datasets, context)

                    if v_acc > curr_best_accuracy_epoch:
                        curr_best_accuracy_epoch = v_acc

                    time_cost = time.time() - start
                    print('Context [{}/{}], Epoch [{}/{}], Step [{}/{}], TrainLoss: {:.4f}; '
                          'TrainAcc: {:.4f}, ValidLoss: {:.4f}; '
                          'Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, AUC: {:.4f}, '
                          'curr_epoch_best_accuracy: {:.4f}, curr_best_accuracy: {:.4f};  Time: {:.2f}s'
                          .format(context, len(train_datasets),
                                  epoch + 1, max_epoch_num, i + 1, total_step, np.mean(tlosses), accuracy, v_loss,
                                  v_acc, v_prec, v_recall, v_f1, auc,
                                  curr_best_accuracy_epoch, curr_best_accuracy, time_cost))

            if curr_best_accuracy_epoch > curr_best_accuracy:
                curr_best_accuracy = curr_best_accuracy_epoch
            else:
                if epoch >= min_epoch_num - 1:
                    # model = copy.deepcopy(best_model)
                    print("best accuracy: {}, early stop!".format(curr_best_accuracy))
                    break
        if args.save:
            if baseline=='joint':
                continue
            elif baseline == 'none':
                test_all_so_far(model, valid_datasets, context, method='none', tofile=True, args=args)
                save_classifier(model, 'none', context, args.index)
            else:
                test_all_so_far(model, valid_datasets, context, method='joint', tofile=True, args=args)
                save_classifier(model, 'joint', context, args.index)


def train_a_batch(model, x, y, scores=None, rnt=0.5, active_classes=None, context=1,
                      **kwargs):
    tlosses = []
    model.train()
    # -however, if some layers are frozen, they should be set to eval() to prevent batch-norm layers from changing
    if model.fcE.frozen:
        model.fcE.eval()
    model.optimizer.zero_grad()
    if x is not None:
        y_hat = model(x)
        if active_classes is not None:
            class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes

            y_hat = y_hat[:, class_entries]


        if model.binaryCE:
            # -binary prediction loss
            binary_targets = lf.to_one_hot(y.cpu(), y_hat.size(1)).to(y.device)
            if model.binaryCE_distill and (scores is not None):
                # -replace targets for previously seen classes with predictions of previous model
                binary_targets[:, :scores.size(1)] = torch.sigmoid(scores / model.KD_temp)
            predL = None if y is None else F.binary_cross_entropy_with_logits(
                input=y_hat, target=binary_targets, reduction='none'
            ).sum(dim=1).mean()  # --> sum over classes, then average over batch
        else:
            # -multiclass prediction loss
            predL = None if y is None else F.cross_entropy(input=y_hat, target=y, reduction='mean')

        # Weigh losses
        loss_cur = predL

        accuracy = None if y is None else (y == y_hat.max(1)[1]).sum().item() / x.size(0)


    loss_total = loss_cur
    tlosses.append(loss_total.item())


    loss_total = loss_cur

    loss_total.backward()

    model.optimizer.step()

    return tlosses, accuracy