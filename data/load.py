import numpy as np
from torchvision import transforms

from .manipulate import SubDataset


def get_dataset(type, dir):
    '''Create [train|valid|test]-dataset.'''

    print(f'reading {type}')
    # datadir = f'{dir}/{type}_old'
    datadir = f'{dir}/{type}'
    print(f'reading {type} done')
    return datadir


def get_context_set(scenario, contexts, classes, seqlen, data_dir, singlehead=False, train_set_per_class=False):

    config = {'classes': classes}

    # check for number of contexts
    # -how many classes per context?
    classes_per_context = (classes // contexts) * 2
    config['classes_per_context'] = classes_per_context
    config['output_units'] = classes
    print('config[output_units]=', config['output_units'])

    # classes = config['classes']

    trainset = get_dataset('train', data_dir)
    validset = get_dataset('valid', data_dir)
    testset = get_dataset('test', data_dir)

    labels_per_dataset_train = [
        list(np.array(range(classes_per_context)) + classes_per_context * context_id) for context_id in range(contexts)
    ]
    print(labels_per_dataset_train)
    labels_per_dataset_valid = [
        list(np.array(range(classes_per_context)) + classes_per_context * context_id) for context_id in range(contexts)
    ]
    labels_per_dataset_test = [
        list(np.array(range(classes_per_context)) + classes_per_context * context_id) for context_id in range(contexts)
    ]

    train_datasets = []
    for labels in labels_per_dataset_train:
        train_datasets.append(SubDataset(trainset, labels, seqlen))
    valid_datasets = []
    for labels in labels_per_dataset_valid:
        valid_datasets.append(SubDataset(validset, labels, seqlen))

    test_datasets = []
    for labels in labels_per_dataset_test:
        test_datasets.append(SubDataset(testset, labels, seqlen))

    return ((train_datasets, valid_datasets, test_datasets), config)


def get_valid_set(scenario, contexts, classes, seqlen, data_dir, singlehead=False, train_set_per_class=False):
    config = {'classes': classes}

    classes_per_context = (classes // contexts) * 2
    config['classes_per_context'] = classes_per_context
    config['output_units'] = classes
    print('config[output_units]=', config['output_units'])

    validset = get_dataset('valid', data_dir)
    labels_per_dataset_valid = [
        list(np.array(range(classes_per_context)) + classes_per_context * context_id) for context_id in range(contexts)
    ]
    valid_datasets = []
    for labels in labels_per_dataset_valid:
        valid_datasets.append(SubDataset(validset, labels, seqlen))

    return valid_datasets, config

def get_test_set(scenario, contexts, classes, seqlen, data_dir, singlehead=False, train_set_per_class=False):
    config = {'classes': classes}

    classes_per_context = (classes // contexts) * 2
    config['classes_per_context'] = classes_per_context
    config['output_units'] = classes
    print('config[output_units]=', config['output_units'])

    testset = get_dataset('test', data_dir)
    labels_per_dataset_test = [
        list(np.array(range(classes_per_context)) + classes_per_context * context_id) for context_id in range(contexts)
    ]
    print('labels_per_dataset_test=', labels_per_dataset_test)
    test_datasets = []
    for labels in labels_per_dataset_test:
        test_datasets.append(SubDataset(testset, labels, seqlen))

    return test_datasets, config