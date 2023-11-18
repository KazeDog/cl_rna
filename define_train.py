from utils import checkattr


def define_trainway(args, model, train_datasets, valid_datasets, baseline):
    if checkattr(args, 'si'):
        from methods.parameter_regularization.si.train import train_cl
        return train_cl(args, model, train_datasets, valid_datasets, args.min_epoch_num,
                        args.max_epoch_num, args.batch_size, baseline)

    elif checkattr(args, 'lwf'):
        from methods.functional_regularization.lwf.train import train_cl
        return train_cl(args, model, train_datasets, valid_datasets, args.min_epoch_num,
                        args.max_epoch_num, args.batch_size, baseline)

    elif checkattr(args, 'er'):
        from methods.repaly.er.train import train_cl
        return train_cl(args, model, train_datasets, valid_datasets, args.min_epoch_num,
                        args.max_epoch_num, args.batch_size, baseline)


    elif checkattr(args, 'dgr'):
        from methods.repaly.dgr.train_vae import train_cl
        return train_cl(args, model, train_datasets, valid_datasets, args.min_epoch_num,
                        args.max_epoch_num, args.batch_size, baseline)
        # from methods.repaly.dgr.train_cwgan import train_cl
        # return train_cl(args, model, train_datasets, valid_datasets, args.min_epoch_num,
        #                 args.max_epoch_num, args.batch_size, baseline)


    elif checkattr(args, 'icarl'):
        from methods.template_based_classification.icarl.train import train_cl
        return train_cl(args, model, train_datasets, valid_datasets, args.min_epoch_num,
                        args.max_epoch_num, args.batch_size, baseline)

    else:
        from methods.baseline.train import train_cl
        return train_cl(args, model, train_datasets, valid_datasets, args.min_epoch_num,
                        args.max_epoch_num, args.batch_size, baseline)
