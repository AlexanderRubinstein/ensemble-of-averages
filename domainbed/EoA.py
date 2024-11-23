# Copyright (c) Salesforce and its affiliates. All Rights Reserved
import json
import numpy as np
import sys
import os
import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(((__file__)))))
from domainbed import datasets
from domainbed import algorithms
from domainbed.lib.fast_data_loader import FastDataLoader
from domainbed import networks
from domainbed import hparams_registry
sys.path.pop(0)

import torch
import torch.nn as nn
import os
import argparse
import time


def append_dict(total_dict, current_dict, allow_new_keys=False):
    """
    Append leaves of possibly nested <current_dict>
    to leaf lists of possibly nested <total_dict>
    """

    def to_create_new_key(is_new_total_dict, allow_new_keys, key, total_dict):
        return is_new_total_dict or (allow_new_keys and key not in total_dict)

    is_new_total_dict = False
    if len(total_dict) == 0:
        is_new_total_dict = True
    for key, value in current_dict.items():
        if isinstance(value, dict):
            if to_create_new_key(
                is_new_total_dict,
                allow_new_keys,
                key,
                total_dict
            ):
                sub_dict = {}
                append_dict(sub_dict, value)
                total_dict[key] = sub_dict
            else:
                assert key in total_dict
                sub_dict = total_dict[key]
                assert isinstance(sub_dict, dict)
                append_dict(sub_dict, value)
                total_dict[key] = sub_dict
        else:
            if to_create_new_key(
                is_new_total_dict,
                allow_new_keys,
                key,
                total_dict
            ):
                total_dict[key] = [value]
            else:
                assert key in total_dict
                assert isinstance(total_dict[key], list)
                total_dict[key].append(value)


def make_values_dict_from_huge_string(huge_string, keys, apply_to_values=None):

    if isinstance(keys, str):
        keys = keys.split()

    huge_string = huge_string.replace('\t', ' ').replace('\n', ' ')

    split = huge_string.split()

    assert len(split) % len(keys) == 0, \
            f"split {split} is not suitable for keys {keys}"

    current_tuple = []
    res = {}
    for item in split:

        current_tuple.append(item)
        if len(current_tuple) < len(keys):
            continue

        for key, value in zip(keys, current_tuple):
            append_dict(res, {key: value}, allow_new_keys=True)

        current_tuple = []

    if apply_to_values is not None:
        for key in res:
            res[key] = np.array(apply_to_values(res[key]))

    return res


class Algorithm(torch.nn.Module):
    def __init__(self, input_shape, hparams, num_classes):
        super(Algorithm, self).__init__()
        self.hparams = hparams
        self.featurizer = networks.Featurizer(input_shape, hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)

        self.featurizer_mo = networks.Featurizer(input_shape, hparams)
        self.classifier_mo = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            hparams['nonlinear_classifier'])

        self.network = self.network.cuda()
        self.network = torch.nn.parallel.DataParallel(self.network).cuda()

        self.network_sma = nn.Sequential(self.featurizer_mo, self.classifier_mo)
        self.network_sma = self.network_sma.cuda()
        self.network_sma = torch.nn.parallel.DataParallel(self.network_sma).cuda()

    def predict(self, x):
        if self.hparams['SMA']:
            return self.network_sma(x)
        else:
            return self.network(x)

def accuracy(models, loader):

    is_hdr = False
    # In HDR case all models are contained in one algorithm
    if len(models) == 1:
        algorithm = models[0]
        network = algorithm.network
        if hasattr(network, "submodels"):
            is_hdr = True
            models = network.submodels

    correct = 0
    total = 0
    weights_offset = 0

    with torch.no_grad():
        for data in tqdm.tqdm(loader):
            x1,y = data[0], data[-1]
            x = x1.cuda()
            y = y.cuda()

            p = None
            for model in models:
                model.to(x.device)
                model.eval()
                if is_hdr:
                    p_i = model(x).detach()
                else:
                    p_i = model.predict(x).detach()
                if p is None:
                    p = p_i
                else:
                    p += p_i

            batch_weights = torch.ones(len(x))

            batch_weights = batch_weights.cuda()
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    return correct / total



def rename_dict(D):
    dnew = {}
    for key, val in D.items():
        pre = key.split('.')[0]
        if pre=='network':
            knew = '.'.join(['network.module'] + key.split('.')[1:])
        else:
            knew = '.'.join(['network_sma.module'] + key.split('.')[1:])
        dnew[knew] = val
    return dnew

def get_test_env_id(path):
    results_path = os.path.join(path, "results.jsonl")
    with open(results_path, "r") as f:
        for j,line in enumerate(f):
            r = json.loads(line[:-1])
            env_id = r['args']['test_envs'][0]
            break
    return env_id

def get_valid_model_selection_paths(path, nenv=4):
    valid_model_id = [[] for _ in range(nenv)]
    listdir = os.listdir(path)
    if 'best_model.pkl' in listdir:
        test_env_id = get_test_env_id(path)
        valid_model_id[test_env_id].append(f'{path}/best_model.pkl')
    else:

        for env in range(nenv):
            cnt=0
            for i, subdir in enumerate(listdir):
                if '.' not in subdir:
                    test_env_id =get_test_env_id(os.path.join(path, subdir))
                    if env==test_env_id:
                        cnt+=1
                        valid_model_id[env].append(f'{path}/{subdir}/best_model.pkl')
    return valid_model_id

def get_ensemble_test_acc(exp_path, nenv, dataset_name, data_dir, hparams, force=False, var=False, file_path=None):

    test_acc = {}

    valid_model_id = get_valid_model_selection_paths(exp_path, nenv=nenv)

    for env in range(nenv):
        # if len(valid_model_id[env]) == 0:
        # 	continue
        if len(valid_model_id[env]) == 0:
            print("Skipping env: ", env)
            continue

        dataset = vars(datasets)[dataset_name](data_dir, [env], hparams)
        assert nenv == len(dataset)
        test_acc[env] = None
        print(f'Test Domain: {dataset.ENVIRONMENTS[env]}')
        data_loader = FastDataLoader(
                dataset=dataset[env],
                batch_size=hparams['batch_size'],# 64*12
                num_workers=hparams['num_workers']) # 64

        # valid_model_id = get_valid_model_selection_paths(exp_path, nenv=len(dataset))

        Algorithm_all = []
        for model_path in valid_model_id[env]:


            algorithm_dict = torch.load(model_path)

            algo_args = algorithm_dict["args"]
            algorithm_name = algo_args["algorithm"]
            if algorithm_name == "HDR":
                train_hparams = hparams_registry.default_hparams(algorithm_name, dataset_name)
                algo_hparams = algo_args["hparams"]
                if isinstance(algo_hparams, str):
                    algo_hparams = json.loads(algo_hparams)
                train_hparams.update(algo_hparams)
                Algorithm_ = algorithms.HDR(
                    dataset.input_shape, dataset.num_classes,
                    len(dataset) - len(algo_args["test_envs"]),
                    hparams=train_hparams
                )
                D = algorithm_dict['model_dict']

            else:
                Algorithm_ = Algorithm(dataset.input_shape, hparams, dataset.num_classes)
                D = rename_dict(algorithm_dict['model_dict'])
            Algorithm_.load_state_dict(D, strict=False)
            Algorithm_all.append(Algorithm_)

        # if len(Algorithm_all) > 0:
        acc = accuracy(Algorithm_all, data_loader)
        print(f'  Test domain Acc: {100.*acc:.2f}%')
        test_acc[env] = acc
        test_acc[env] = (acc, dataset.ENVIRONMENTS[env])

    return test_acc


def eval(path_with_checkpoints, nenv, dataset_name, data_dir, hparams):
    tic = time.time()
    test_acc = get_ensemble_test_acc(path_with_checkpoints, nenv, dataset_name, data_dir, hparams, force=False)
    test_acc = {k: (float(f'{100.*v[0]:.1f}'), v[1:]) for k, v in test_acc.items()}
    toc = time.time()
    all_accs = [val[0] for val in test_acc.values()]
    print(f'Avg: {np.array(all_accs).mean():.1f}, Time taken: {toc-tic:.2f}s')
    return test_acc


parser = argparse.ArgumentParser(description='Ensemble of Averages')
parser.add_argument('--data_dir', type=str)
parser.add_argument('--dataset', type=str, default="PACS")
parser.add_argument('--arch', type=str, default="resnet50")
parser.add_argument('--output_dir', type=str, help='the experiment directory where the results of domainbed.scripts.sweep were saved')
parser.add_argument('--hparams', type=str, help='JSON-serialized hparams dict')
parser.add_argument('--parsing_keys', type=str, help='keys in huge string')
parser.add_argument('--paths_key', type=str, help='key of checkpoint folder in huge string')
parser.add_argument('--result_savepath', type=str, help='path to save results to')

args = parser.parse_args()

dataset_name= args.dataset
if dataset_name in ['PACS', 'TerraIncognita', 'VLCS', 'OfficeHome']:
    nenv = 4
elif dataset_name=='DomainNet':
    nenv = 6

data_dir= args.data_dir
hparams = {'data_augmentation': False, "nonlinear_classifier": False, "resnet_dropout": 0, "arch": args.arch, "batch_size": 64, "num_workers":1}
args_hparams = json.loads(args.hparams)
if args_hparams['SMA'] in ["True", "true"]:
    args_hparams['SMA'] = True
else:
    args_hparams['SMA'] = False
if args.hparams:
    hparams.update(args_hparams)

if args.parsing_keys is None:
    path_with_checkpoints = args.output_dir
    eval(path_with_checkpoints, nenv, dataset_name, data_dir, hparams)
else:
    assert args.paths_key is not None
    assert args.result_savepath is not None
    if os.path.exists(args.result_savepath):
        all_evals_acc = torch.load(args.result_savepath)
    else:
        os.makedirs(os.path.dirname(args.result_savepath), exist_ok=True)
        all_evals_acc = {}
    paths_with_checkpoints = make_values_dict_from_huge_string(
        # huge_string, keys,
        huge_string=args.output_dir,
        keys=args.parsing_keys.split(","),
        # apply_to_values=json.loads
    )

    # for key in args.parsing_keys:
    varied_hp = paths_with_checkpoints
    paths_with_checkpoints = varied_hp.pop(args.paths_key)

    # for path_with_checkpoints, varied_hp in zip(paths_with_checkpoints, varied_hps):
    for i in range(len(paths_with_checkpoints)):
        path_with_checkpoints = paths_with_checkpoints[i]
        result_id = os.path.basename(path_with_checkpoints)
        for key, value in varied_hp.items():
            result_id += f'_{key}_{value[i]}'
        if result_id not in all_evals_acc:
            print(f'Evaluating {result_id}')
            eval_acc = eval(path_with_checkpoints, nenv, dataset_name, data_dir, hparams)
            all_evals_acc[result_id] = eval_acc
            torch.save(all_evals_acc, args.result_savepath)

# eval??


# tic = time.time()
# test_acc = get_ensemble_test_acc(path, nenv, dataset_name, data_dir, hparams, force=False)
# test_acc = {k: float(f'{100.*test_acc[k]:.1f}') for k in test_acc.keys()}
# toc = time.time()
# print(f'Avg: {np.array(list(test_acc.values())).mean():.1f}, Time taken: {toc-tic:.2f}s')
