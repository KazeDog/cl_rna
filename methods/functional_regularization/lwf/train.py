import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
import time
import copy

from torch.utils.data import ConcatDataset

from data.manipulate import SubDataset, MemorySetDataset
from evaluate import test_all_so_far
from save_model import save_classifier
from utils import get_data_loader, checkattr
import model_utils.loss_functions as lf


def train_cl(args, model, train_datasets, valid_datasets, min_epoch_num, max_epoch_num, batch_size=32,
             baseline='none', **kwargs):


    model.train()

    # Use cuda?
    cuda = model._is_on_cuda()
    device = model._device()

    # Initiate possible sources for replay (no replay for 1st context)
    ReplayStoredData = ReplayGeneratedData = ReplayCurrentData = False
    previous_model = None

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
        if baseline=="cummulative" and per_context:
            ReplayStoredData = True
            previous_datasets = train_datasets

        print('baseline=', baseline)

        training_dataset = train_dataset

        active_classes = list(range(context * int(model.classes_per_context / 2) + 1))
        print('active_classes=', active_classes)
        # Reset state of optimizer(s) for every context (if requested)
        print('model.optim_type=', model.optim_type)
        if model.optim_type=="adam_reset":
            model.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))

        # Initialize # iters left on current data-loader(s)
        iters_left = iters_left_previous = 1
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
        for epoch in range(max_epoch_num):
            curr_best_accuracy_epoch = 0

            # -----------------Collect data------------------#
            for i, sfeatures in enumerate(data_loader):
                if ReplayStoredData:
                    if per_context:
                        up_to_context = context if baseline == "cummulative" else context - 1
                        batch_size_replay = int(np.ceil(batch_size / up_to_context)) if (
                                    up_to_context > 1) else batch_size
                        # -if different active classes per context (e.g., Task-IL), need separate replay for each context
                        for context_id in range(up_to_context):
                            batch_size_to_use = min(batch_size_replay, len(previous_datasets[context_id]))
                            iters_left_previous[context_id] -= 1
                            if iters_left_previous[context_id] == 0:
                                data_loader_previous[context_id] = iter(get_data_loader(
                                    previous_datasets[context_id], batch_size_to_use, cuda=cuda, drop_last=True
                                ))
                                iters_left_previous[context_id] = len(data_loader_previous[context_id])
                    else:
                        iters_left_previous -= 1
                        if iters_left_previous == 0:
                            batch_size_to_use = min(batch_size, len(ConcatDataset(previous_datasets)))
                            data_loader_previous = iter(get_data_loader(ConcatDataset(previous_datasets),
                                                                        batch_size_to_use, cuda=cuda, drop_last=True))
                            iters_left_previous = len(data_loader_previous)

                #####-----CURRENT BATCH-----#####
                if baseline=="cummulative" and per_context:
                    x = y = scores = None
                else:
                    x, y = sfeatures
                    if per_context and not per_context_singlehead:
                        y = y.cpu().numpy().tolist()
                        y = [label - context + 1 if label != 0 else label for label in y]
                        y = torch.LongTensor(y)
                    # --> adjust the y-targets to the 'active range'
                    x, y = x.to(device), y.to(device)                    #--> transfer them to correct device
                    # If --bce & --bce-distill, calculate scores for past classes of current batch with previous model
                    binary_distillation = hasattr(model, "binaryCE") and model.binaryCE and model.binaryCE_distill
                    if binary_distillation and model.scenario in ("class", "all") and (previous_model is not None):
                        with torch.no_grad():
                            scores = previous_model.classify(
                                x, no_prototypes=True
                            )[:, :(model.classes_per_context * (context - 1))]
                    else:
                        scores = None


                #####-----REPLAYED BATCH-----#####
                if not ReplayStoredData and not ReplayGeneratedData and not ReplayCurrentData:
                    x_ = y_ = scores_ = context_used = None   #-> if no replay

                if ReplayCurrentData:
                    x_ = x  # --> use current context inputs
                    context_used = None
                    # Get target scores and labels (i.e., [scores_] / [y_]) -- using previous model, with no_grad()
                    if not per_context:
                        # -if replay does not need to be evaluated separately for each context
                        with torch.no_grad():
                            scores_ = previous_model.classify(x_, no_prototypes=True)
                        if model.scenario == "class" and model.neg_samples == "all-so-far":
                            scores_ = scores_[:, :(context)]
                            # -> if [scores_] is not same length as [x_], zero probs are added in [loss_fn_kd]-function
                        # -also get the 'hard target'
                        _, y_ = torch.max(scores_, dim=1)
                    else:
                        # -[x_] needs to be evaluated according to each past context, so make list with entry per context
                        scores_ = list()
                        y_ = list()
                        # -if no context-mask and no conditional generator, all scores can be calculated in one go
                        if previous_model.mask_dict is None and not type(x_) == list:
                            with torch.no_grad():
                                all_scores_ = previous_model.classify(x_, no_prototypes=True)
                        for context_id in range(context - 1):
                            # -if there is a context-mask (i.e., XdG), obtain predicted scores for each context separately
                            if previous_model.mask_dict is not None:
                                previous_model.apply_XdGmask(context=context_id + 1)
                            if previous_model.mask_dict is not None or type(x_) == list:
                                with torch.no_grad():
                                    all_scores_ = previous_model.classify(x_[context_id] if type(x_) == list else x_,
                                                                          no_prototypes=True)
                            temp_scores_ = all_scores_
                            if active_classes is not None:
                                temp_scores_ = temp_scores_[:, active_classes[context_id]]
                            scores_.append(temp_scores_)
                            # - also get hard target
                            _, temp_y_ = torch.max(temp_scores_, dim=1)
                            y_.append(temp_y_)

                    # Only keep predicted y/scores if required (as otherwise unnecessary computations will be done)
                    y_ = y_ if (model.replay_targets == "hard") else None
                    scores_ = scores_ if (model.replay_targets == "soft") else None


                tlosses, accuracy = train_a_batch(model, x, y, x_=x_, y_=y_, scores=scores, scores_=scores_, rnt = 1./context,
                                                contexts_=context_used, active_classes=active_classes, context=context)

                if i == total_step - 1:
                    v_loss, v_acc, v_prec, v_recall, v_f1, auc = test_all_so_far(model, valid_datasets, context)

                    if v_acc > curr_best_accuracy_epoch:
                        curr_best_accuracy_epoch = v_acc

                    time_cost = time.time() - start
                    print('<CLASSIFIER> | Context [{}/{}], Epoch [{}/{}], Step [{}/{}], TrainLoss: {:.4f}; '
                          'TrainAcc: {:.4f}, ValidLoss: {:.4f}; '
                          'Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, AUC: {:.4f}, '
                          'curr_epoch_best_accuracy: {:.4f}, curr_best_accuracy: {:.4f}; Time: {:.2f}s'
                          .format(context, len(train_datasets),
                                  epoch + 1, max_epoch_num, i + 1, total_step, np.mean(tlosses), accuracy, v_loss,
                                  v_acc, v_prec, v_recall, v_f1, auc,
                                  curr_best_accuracy_epoch, curr_best_accuracy, time_cost))

            if curr_best_accuracy_epoch > curr_best_accuracy:
                curr_best_accuracy = curr_best_accuracy_epoch
            else:
                if epoch >= min_epoch_num - 1:
                    print("best accuracy: {}, early stop!".format(curr_best_accuracy))
                    break
        if args.save:
            test_all_so_far(model, valid_datasets, context, method='lwf', tofile=True, args=args)
            save_classifier(model, 'lwf', context, args.index)

        # REPLAY: update source for replay
        if context<len(train_datasets) and hasattr(model, 'replay_mode'):
            previous_model = copy.deepcopy(model).eval()
            ReplayCurrentData = True

def train_a_batch(model, x, y, scores=None, x_=None, y_=None, scores_=None, rnt=0.5, active_classes=None, context=1,
                      **kwargs):
    tlosses = []
    model.train()
    if model.fcE.frozen:
        model.fcE.eval()
    model.optimizer.zero_grad()
    # Should gradient be computed separately for each context? (needed when a context-mask is combined with replay)
    gradient_per_context = True if ((model.mask_dict is not None) and (x_ is not None)) else False

    if x_ is not None:
        PerContext = (type(y_) == list) if (y_ is not None) else (type(scores_) == list)
        if not PerContext:
            y_ = [y_]
            scores_ = [scores_]
            if active_classes is not None:
                if type(active_classes[0]) is not list:
                    active_classes = [active_classes]
                else:
                    active_classes = active_classes
            else:
                active_classes = None
        n_replays = len(y_) if (y_ is not None) else len(scores_)

        # Prepare lists to store losses for each replay
        loss_replay = [None] * n_replays
        predL_r = [None] * n_replays
        distilL_r = [None] * n_replays

        # Run model (if [x_] is not a list with separate replay per context and there is no context-specific mask)
        if (not type(x_) == list) and (model.mask_dict is None):
            y_hat_all = model(x_)

        # Loop to evalute predictions on replay according to each previous context
        for replay_id in range(n_replays):
            # print('replay_id=', replay_id)
            # -if [x_] is a list with separate replay per context, evaluate model on this context's replay
            if (type(x_) == list) or (model.mask_dict is not None):
                x_temp_ = x_[replay_id] if type(x_) == list else x_
                if model.mask_dict is not None:
                    model.apply_XdGmask(context=replay_id + 1)
                y_hat_all = model(x_temp_)

            # -if needed, remove predictions for classes not active in the replayed context
            y_hat = y_hat_all if (active_classes is None) else y_hat_all[:, active_classes[replay_id]]

            # Calculate losses
            if (y_ is not None) and (y_[replay_id] is not None):
                if model.binaryCE:
                    binary_targets_ = lf.to_one_hot(y_[replay_id].cpu(), y_hat.size(1)).to(
                        y_[replay_id].device)
                    predL_r[replay_id] = F.binary_cross_entropy_with_logits(
                        input=y_hat, target=binary_targets_, reduction='none'
                    ).sum(dim=1).mean()  # --> sum over classes, then average over batch
                else:
                    predL_r[replay_id] = F.cross_entropy(y_hat, y_[replay_id], reduction='mean')
            if (scores_ is not None) and (scores_[replay_id] is not None):
                # n_classes_to_consider = scores.size(1) #--> with this version, no zeroes are added to [scores]!
                n_classes_to_consider = y_hat.size(
                    1)  # --> zeros will be added to [scores] to make it this size!
                kd_fn = lf.loss_fn_kd_binary if model.binaryCE else lf.loss_fn_kd
                distilL_r[replay_id] = kd_fn(scores=y_hat[:, :n_classes_to_consider],
                                             target_scores=scores_[replay_id], T=model.KD_temp)

            # Weigh losses
            if model.replay_targets == "hard":
                loss_replay[replay_id] = predL_r[replay_id]
            elif model.replay_targets == "soft":
                loss_replay[replay_id] = distilL_r[replay_id]

            # If needed, perform backward pass before next context-mask (gradients of all contexts will be accumulated)
            if gradient_per_context:
                weight = 1. if model.use_replay == 'inequality' else (1. - rnt)
                weighted_replay_loss_this_context = weight * loss_replay[replay_id] / n_replays
                weighted_replay_loss_this_context.backward()

    # Calculate total replay loss
    loss_replay = None if (x_ is None) else sum(loss_replay) / n_replays
    if (x_ is not None) and model.lwf_weighting and (not model.scenario == 'class'):
        loss_replay *= (context - 1)

    # If using the replayed loss as an inequality constraint, calculate and store averaged gradient of replayed data
    if model.use_replay in ('inequality', 'both') and x_ is not None:
        # Perform backward pass to calculate gradient of replayed batch (if not yet done)
        if not gradient_per_context:
            if model.use_replay == 'both':
                loss_replay = (1 - rnt) * loss_replay
            loss_replay.backward()
        # Reorganize the gradient of the replayed batch as a single vector
        grad_rep = []
        for p in model.parameters():
            if p.requires_grad:
                grad_rep.append(p.grad.data.view(-1))
        grad_rep = torch.cat(grad_rep)
        # If gradients are only used as inequality constraint, reset them
        if model.use_replay == 'inequality':
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

        # Calculate training-accuracy
        accuracy = None if y is None else (y == y_hat.max(1)[1]).sum().item() / x.size(0)
    else:
        accuracy = predL = None
        # -> it's possible there is only "replay" [i.e., for offline with incremental context learning]

    if x_ is None:
        loss_total = loss_cur
    else:
        loss_total = loss_replay if (x is None) else rnt * loss_cur + (1 - rnt) * loss_replay
    tlosses.append(loss_total.item())

    loss_total.backward()

    model.optimizer.step()

    return tlosses, accuracy


