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
import define_models

def define_vae(args, device):
    from methods.repaly.dgr.vae import VAE
    generator = VAE(
        image_size=args.seqlen,
        image_channels=1,
    ).to(device)
    # -initialize parameters
    define_models.init_params(generator, args)
    # -set optimizer(s)
    generator.optim_list = [{'params': filter(lambda p: p.requires_grad, generator.parameters()),
                             'lr': args.lr}]
    print('generator.optim_list=', generator.optim_list)
    generator.optim_type = args.optimizer
    if generator.optim_type in ("adam", "adam_reset"):
        generator.optimizer = optim.Adam(generator.optim_list, betas=(0.9, 0.999))
    elif generator.optim_type == "sgd":
        generator.optimizer = optim.SGD(generator.optim_list)

    return generator

def train_cl(args, model, train_datasets, valid_datasets, min_epoch_num, max_epoch_num, batch_size=32,
             baseline='none', gen_epoch_num=100, **kwargs):


    # Set model in training-mode
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

        print('baseline=', baseline)

        training_dataset = train_dataset

        generator = define_vae(args, device)

        active_classes = list(range(context * int(model.classes_per_context / 2) + 1))
        print('active_classes=', active_classes)
        # Reset state of optimizer(s) for every context (if requested)
        print('model.optim_type=', model.optim_type)
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
                    x, y = sfeatures
                    if per_context and not per_context_singlehead:
                        y = y.cpu().numpy().tolist()
                        y = [label - context + 1 if label != 0 else label for label in y]
                        y = torch.LongTensor(y)
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

                if ReplayGeneratedData:
                    # -which classes are allowed to be generated? (relevant if conditional generator / decoder-gates)
                    if context == 1:
                        allowed_classes = []
                    else:
                        allowed_classes = list(range(context))
                    # -which contexts are allowed to be generated? (only relevant if "Domain-IL" with context-gates)
                    allowed_domains = list(range(context - 1))
                    # -generate inputs representative of previous contexts
                    x_temp_ = previous_generator.sample(batch_size, allowed_classes=allowed_classes,
                                                        allowed_domains=allowed_domains, only_x=False)
                    x_ = x_temp_[0] if type(x_temp_) == tuple else x_temp_
                    context_used = x_temp_[2] if type(x_temp_) == tuple else None

                # ---OUTPUTS---#
                if ReplayGeneratedData or ReplayCurrentData:
                    # Get target scores and labels (i.e., [scores_] / [y_]) -- using previous model, with no_grad()
                    if not per_context:
                        # -if replay does not need to be evaluated separately for each context
                        with torch.no_grad():
                            scores_ = previous_model.classify(x_, no_prototypes=True)
                            scores_ = scores_[:, :context]
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
                                    all_scores_ = previous_model.classify(
                                        x_[context_id] if type(x_) == list else x_,
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
                          'curr_epoch_best_accuracy: {:.4f}; Time: {:.2f}s'
                          .format(context, len(train_datasets),
                                  epoch + 1, max_epoch_num, i + 1, total_step, np.mean(tlosses), accuracy, v_loss,
                                  v_acc, v_prec, v_recall, v_f1, auc,
                                  curr_best_accuracy_epoch, time_cost))


            if curr_best_accuracy_epoch > curr_best_accuracy:
                curr_best_accuracy = curr_best_accuracy_epoch
            else:
                if epoch >= min_epoch_num - 1:
                    print("best accuracy: {}, early stop!".format(curr_best_accuracy))
                    break

        if args.save:
            test_all_so_far(model, valid_datasets, context, method='dgr', tofile=True, args=args)
            save_classifier(model, 'dgr', context, args.index)

        if context < len(train_datasets):
            for epoch in range(args.gen_epoch):
                for i, sfeatures in enumerate(data_loader):
                    if baseline == "cummulative" and per_context:
                        x = y = scores = None
                    else:
                        x, y = sfeatures
                        if per_context and not per_context_singlehead:
                            y = y.cpu().numpy().tolist()
                            y = [label - context + 1 if label != 0 else label for label in y]
                            y = torch.LongTensor(y)
                        # --> adjust the y-targets to the 'active range'
                        x, y = x.to(device), y.to(device)  # --> transfer them to correct device


                    #####-----REPLAYED BATCH-----#####
                    if not ReplayStoredData and not ReplayGeneratedData and not ReplayCurrentData:
                        x_ = y_ = scores_ = context_used = None  # -> if no replay

                    if ReplayGeneratedData:
                        # -which classes are allowed to be generated? (relevant if conditional generator / decoder-gates)
                        if model.scenario == "domain":
                            allowed_classes = None
                        elif context == 1:
                            allowed_classes = []
                        else:
                            allowed_classes = list(range(context))
                        # -which contexts are allowed to be generated? (only relevant if "Domain-IL" with context-gates)
                        allowed_domains = list(range(context - 1))
                        # -generate inputs representative of previous contexts
                        x_temp_ = previous_generator.sample(batch_size, allowed_classes=allowed_classes,
                                                            allowed_domains=allowed_domains, only_x=False)
                        x_ = x_temp_[0] if type(x_temp_) == tuple else x_temp_
                        context_used = x_temp_[2] if type(x_temp_) == tuple else None

                    # ---OUTPUTS---#
                    if ReplayGeneratedData or ReplayCurrentData:
                        # Get target scores and labels (i.e., [scores_] / [y_]) -- using previous model, with no_grad()
                        if not per_context:
                            # -if replay does not need to be evaluated separately for each context
                            with torch.no_grad():
                                scores_ = previous_model.classify(x_, no_prototypes=True)
                            if model.scenario == "class" and model.neg_samples == "all-so-far":
                                scores_ = scores_[:, :context]
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
                                        all_scores_ = previous_model.classify(
                                            x_[context_id] if type(x_) == list else x_,
                                            no_prototypes=True)
                                temp_scores_ = all_scores_
                                if active_classes is not None:
                                    temp_scores_ = temp_scores_[:, active_classes[context_id]]
                                scores_.append(temp_scores_)
                                # - also get hard target
                                _, temp_y_ = torch.max(temp_scores_, dim=1)
                                y_.append(temp_y_)

                    loss_dict = train_a_batch_gen(generator, x, x_=x_, rnt=1. / context)

                    if i == total_step - 1:
                        time_cost = time.time() - start
                        print('<VAE> | Context [{}/{}], Epoch [{}/{}], Step [{}/{}], TrainLoss: {:.4f}; Time: {:.2f}s'
                              .format(context, len(train_datasets),
                                      epoch + 1, gen_epoch_num, i + 1, total_step, loss_dict['loss_total'], time_cost)
                              )

        # REPLAY: update source for replay
        if context < len(train_datasets) and hasattr(model, 'replay_mode'):
            previous_model = copy.deepcopy(model).eval()
            print('model.replay_mode=', model.replay_mode)
            if model.replay_mode == 'generative':
                ReplayGeneratedData = True
                previous_generator = copy.deepcopy(generator).eval() if generator is not None else previous_model

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
        # If there are different predictions per context, [y_] or [scores_] are lists and [x_] must be evaluated
        # separately on each of them (although [x_] could be a list as well!)
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
            # active_classes = [active_classes] if (active_classes is not None) else None
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
            predL = None if y is None else F.cross_entropy(input=y_hat, target=y, reduction='mean')

        # Weigh losses
        loss_cur = predL

        # Calculate training-accuracy
        accuracy = None if y is None else (y == y_hat.max(1)[1]).sum().item() / x.size(0)
    else:
        accuracy = predL = None
    if x_ is None:
        loss_total = loss_cur
    else:
        loss_total = loss_replay if (x is None) else rnt * loss_cur + (1 - rnt) * loss_replay

    tlosses.append(loss_total.item())

    loss_total.backward()

    model.optimizer.step()

    return tlosses, accuracy

def train_a_batch_gen(generator, x, x_=None, rnt=0.5, **kwargs):
    '''Train model for one batch ([x]), possibly supplemented with replayed data ([x_]).

    [x]                 <tensor> batch of inputs (could be None, in which case only 'replayed' data is used)
    [x_]                None or (<list> of) <tensor> batch of replayed inputs
    [rnt]               <number> in [0,1], relative importance of new context
    '''

    # Set model to training-mode
    generator.train()
    # -however, if some layers are frozen, they should be set to eval() to prevent batch-norm layers from changing
    if generator.fcE.frozen:
        generator.fcE.eval()

    # Reset optimizer
    generator.optimizer.zero_grad()

    ##--(1)-- CURRENT DATA --##
    if x is not None:
        # Run the model
        recon_batch, mu, logvar, z = generator(x, full=True, reparameterize=True)

        # Calculate losses
        reconL, variatL = generator.loss_function(x=x, x_recon=recon_batch, mu=mu, z=z, logvar=logvar)

        # Weigh losses as requested
        loss_cur = generator.lamda_rcl*reconL + generator.lamda_vl*variatL

    ##--(2)-- REPLAYED DATA --##
    if x_ is not None:
        # Run the model
        recon_batch, mu, logvar, z = generator(x_, full=True, reparameterize=True)

        # Calculate losses
        reconL_r, variatL_r = generator.loss_function(x=x_, x_recon=recon_batch, mu=mu, z=z, logvar=logvar)

        # Weigh losses as requested
        loss_replay = generator.lamda_rcl*reconL_r + generator.lamda_vl*variatL_r

    # Calculate total loss
    loss_total = loss_replay if (x is None) else (loss_cur if x_ is None else rnt*loss_cur+(1-rnt)*loss_replay)

    # Backpropagate errors
    loss_total.backward()
    # Take optimization-step
    generator.optimizer.step()

    # Return the dictionary with different training-loss split in categories
    return {
        'loss_total': loss_total.item(),
        'recon': reconL.item() if x is not None else 0,
        'variat': variatL.item() if x is not None else 0,
        'recon_r': reconL_r.item() if x_ is not None else 0,
        'variat_r': variatL_r.item() if x_ is not None else 0,
    }



