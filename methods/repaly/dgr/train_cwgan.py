import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
import time
import copy

from torch.utils.data import ConcatDataset

from data.manipulate import SubDataset, MemorySetDataset
from evaluate import test_all_so_far
from save_model import save_classifier, save_gan
from utils import get_data_loader, checkattr, to_onehot
import model_utils.loss_functions as lf
import define_models
from .cwgan import Generator, Discriminator

def define_gan(args, classes, device):

    generator = Generator(
        channels_noise=args.z_dim,
        image_channels=1,
        features_g=args.seqlen,
        classes=classes
    ).to(device)
    discriminator = Discriminator(
        image_channels=1,
        features_d=args.seqlen,
        classes=classes
    ).to(device)

    define_models.initialize_weights(generator)
    # -set optimizer(s)
    generator.optim_list = [{'params': filter(lambda p: p.requires_grad, generator.parameters()),
                             'lr': args.lr}]
    print('generator.optim_list=', generator.optim_list)
    generator.optim_type = args.optimizer
    if generator.optim_type in ("adam", "adam_reset"):
        generator.optimizer = optim.Adam(generator.optim_list, betas=(0.9, 0.999))
    elif generator.optim_type == "sgd":
        generator.optimizer = optim.SGD(generator.optim_list)

    define_models.initialize_weights(discriminator)
    # -set optimizer(s)
    discriminator.optim_list = [{'params': filter(lambda p: p.requires_grad, discriminator.parameters()),
                                 'lr': args.lr}]
    discriminator.optim_type = args.optimizer
    if discriminator.optim_type in ("adam", "adam_reset"):
        discriminator.optimizer = optim.Adam(discriminator.optim_list, betas=(0.9, 0.999))
    elif discriminator.optim_type == "sgd":
        discriminator.optimizer = optim.SGD(discriminator.optim_list)

    return generator, discriminator



def train_cl(args, model, train_datasets, valid_datasets, min_epoch_num, max_epoch_num,
             batch_size=32, baseline='none', **kwargs):

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
        if baseline == 'joint':
            if context < len(train_datasets):
                continue
            else:
                baseline = "cummulative"

        # If using the "cummulative" (or "joint") baseline, create a large training dataset of all contexts so far
        if baseline == "cummulative" and (not per_context):
            train_dataset = ConcatDataset(train_datasets[:context])
        # -but if "cummulative"+[per_context]: training on each context must be separate, as a trick to achieve this,
        #                                      all contexts so far are treated as replay (& there is no current batch)
        if baseline == "cummulative" and per_context:
            ReplayStoredData = True

        print('baseline=', baseline)

        training_dataset = train_dataset


        generator, discriminator = define_gan(args, context + 1, device)

        active_classes = list(range(context * int(model.classes_per_context / 2) + 1))
        print('active_classes=', active_classes)
        # Reset state of optimizer(s) for every context (if requested)
        print('model.optim_type=', model.optim_type)
        if model.optim_type == "adam_reset":
            model.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))

        print('per_context=', per_context)
        if per_context:
            up_to_context = context if baseline == "cummulative" else context - 1
            print('up_to_context=', up_to_context)
            iters_left_previous = [1] * up_to_context
            print('iters_left_previous=', iters_left_previous)
            data_loader_previous = [None] * up_to_context
            print('data_loader_previous=', data_loader_previous)

        data_loader = get_data_loader(training_dataset, batch_size, cuda=cuda, drop_last=True)
        total_step = len(data_loader)
        print("total_step: {}".format(total_step))
        curr_best_accuracy = 0
        iters = max_epoch_num * len(data_loader)
        for epoch in range(max_epoch_num):
            curr_best_accuracy_epoch = 0
            tlosses = []

            # -----------------Collect data------------------#
            for i, sfeatures in enumerate(data_loader):

                #####-----CURRENT BATCH-----#####
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
                    x_ = y_ = scores_ = context_used = None  # -> if no replay

                # print('ReplayGeneratedData=', ReplayGeneratedData)
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
                                                        z_dim=args.z_dim)
                    x_ = x_temp_[0]
                    y_ = x_temp_[1]

                if ReplayGeneratedData or ReplayCurrentData:
                    # Get target scores and labels (i.e., [scores_] / [y_]) -- using previous model, with no_grad()

                    # Only keep predicted y/scores if required (as otherwise unnecessary computations will be done)
                    y_ = y_ if (model.replay_targets == "hard") else None
                    scores_ = scores_ if (model.replay_targets == "soft") else None
                loss, accuracy = train_a_batch(model, x, y, x_=x_, y_=y_, scores=scores, scores_=scores_,
                                                  rnt=1. / context,
                                                  contexts_=context_used, active_classes=active_classes,
                                                  context=context)
                tlosses.append(loss)
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

                    ##-->> Generative / Current Replay <<--##

                    # ---INPUTS---#
                    if ReplayCurrentData:
                        x_ = x  # --> use current context inputs
                        context_used = None

                    if ReplayGeneratedData:
                        # -which classes are allowed to be generated? (relevant if conditional generator / decoder-gates)
                        if model.scenario == "domain":
                            allowed_classes = None
                        elif context == 1:
                            allowed_classes = []
                        else:
                            allowed_classes = list(range(context))
                        x_temp_ = previous_generator.sample(batch_size, allowed_classes=allowed_classes,
                                                            z_dim=args.z_dim)
                        x_ = x_temp_[0] if type(x_temp_) == tuple else x_temp_
                        y_ = x_temp_[1]

                    # ---OUTPUTS---#
                    if ReplayGeneratedData or ReplayCurrentData:

                        # Only keep predicted y/scores if required (as otherwise unnecessary computations will be done)
                        y_ = y_ if (model.replay_targets == "hard") else None
                        scores_ = scores_ if (model.replay_targets == "soft") else None

                    loss_dict = train_a_batch_gen_ununion(generator, discriminator, batch_size, x, y, x_=x_, y_=y_,
                                                  z_dim=args.z_dim, device=device, rnt=1./context, d_iters=args.disc_epoch)

                    if i == total_step - 1:
                        time_cost = time.time() - start
                        print('<GAN> | Context [{}/{}], Epoch [{}/{}], Step [{}/{}], DiscLoss: {:.4f}, GenLoss: {:.4f}, '
                              'Time: {:.2f}s'
                              .format(context, len(train_datasets),
                                      epoch + 1, args.gen_epoch, i + 1, total_step, loss_dict['loss_disc'],
                                      loss_dict['loss_gen'], time_cost)
                              )
        if context < len(train_datasets) and hasattr(model, 'replay_mode'):
            previous_model = copy.deepcopy(model).eval()
            print('model.replay_mode=', model.replay_mode)
            if model.replay_mode == 'generative':
                ReplayGeneratedData = True
                previous_generator = copy.deepcopy(generator).eval() if generator is not None else previous_model


def train_a_batch(model, x, y, scores=None, x_=None, y_=None, scores_=None, rnt=0.5, active_classes=None, context=1,
                  **kwargs):
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

    loss_total.backward()

    model.optimizer.step()

    return loss_total.item(), accuracy


def gradient_penalty(critic, label, real, fake, device="cpu"):
    real = real.reshape(-1, 1, real.shape[1], real.shape[2])
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, label)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

def train_a_batch_gen_ununion(generator, discriminator, batch_size, x, y, x_, y_, z_dim, device, rnt=0.5, d_iters=5):
    LAMBDA_GP = 10

    generator.train()
    discriminator.train()
    x = F.one_hot(x, 5)
    if x_ is not None:
        x_ = F.one_hot(x_, 5)
    for _ in range(d_iters):
        noise_current = torch.randn(x.shape[0], z_dim, 1, 1).to(device)
        fake_current = generator(noise_current, y)
        disc_real_current = discriminator(x, y).reshape(-1)
        disc_fake_current = discriminator(fake_current, y).reshape(-1)

        gp_current = gradient_penalty(discriminator, y, x, fake_current, device=device)
        loss_disc_current = (
                -(torch.mean(disc_real_current) - torch.mean(disc_fake_current)) + LAMBDA_GP * gp_current
        )
        if x_ is not None:
            noise_replay = torch.randn(x_.shape[0], z_dim, 1, 1).to(device)
            fake_replay = generator(noise_replay, y_)
            disc_real_replay = discriminator(x_, y_).reshape(-1)
            disc_fake_replay = discriminator(fake_replay, y_).reshape(-1)

            gp_replay = gradient_penalty(discriminator, y_, x_, fake_replay, device=device)
            loss_disc_replay = (
                    -(torch.mean(disc_real_replay) - torch.mean(disc_fake_replay)) + LAMBDA_GP * gp_replay
            )
            loss_disc = rnt * loss_disc_current + (1-rnt) * loss_disc_replay
        else:
            loss_disc = loss_disc_current

        discriminator.zero_grad()
        loss_disc.backward(retain_graph=True)
        discriminator.optimizer.step()

    gen_fake = discriminator(fake_current, y).reshape(-1)

    loss_gen = -torch.mean(gen_fake)
    generator.zero_grad()
    loss_gen.backward()
    generator.optimizer.step()

    return {
        'loss_disc': loss_disc,
        'loss_gen': loss_gen,
    }
