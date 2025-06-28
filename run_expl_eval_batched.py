import os, sys
import random
import string
import time, datetime
import itertools
import traceback
from functools import reduce

import joblib
import joblib as jl

import scipy.stats
from scipy.optimize import linprog
import sklearn.metrics
from sklearn.linear_model import LassoLars
from sklearn.manifold import TSNE
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.data import DataLoader, Data
import numpy as np

import matplotlib.pyplot as plt

from model.Twitter.BiGCN_Twitter import BiGCN
from model.Twitter.BiGAT_Twitter import CHGAT
from model.Twitter.EBGCN import EBGCN
from Process.process import loadBiData, loadTree
from Process.pheme9fold import load9foldData
from Process.rand5fold import load5foldData
from Process.getPHEMEgraph import getRawData

from tools.earlystopping import EarlyStopping
from tools.evaluate import *

# from torch_geometric.utils import add_remaining_self_loops
# from torch_geometric.utils.num_nodes import maybe_num_nodes

import lrp_pytorch.modules.utils as lrp_utils
from lrp_pytorch.modules.base import safe_divide
from tqdm import tqdm
import copy
import argparse
import json

import nltk
from nltk.corpus import stopwords

from transformers import BertTokenizer, BertModel
# from interpret_bert.interpret_nlp.visualization.heatmap import html_heatmap
# from IPython.core.display import display, HTML

# nltk.download('stopwords')
STOPWORDS = {}
for word in stopwords.words('english'):
    STOPWORDS[word] = 0
FOLD_2_EVENTNAME = {0: 'charliehebdo',
                    1: 'ebola',
                    2: 'ferguson',
                    3: 'germanwings',
                    4: 'gurlitt',
                    5: 'ottawashooting',
                    6: 'prince',
                    7: 'putinmissing',
                    8: 'sydneysiege'}
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
EXPLAIN_DIR = os.path.join(DATA_DIR, 'explain')
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'model', 'Twitter', 'checkpoints')
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.enabled = False

LAYERS = ['input', 'td_conv1', 'td_conv2', 'bu_conv2', 'bu_conv2']

LRP_PARAMS = {
    'linear_eps': 1e-6,
    'gcn_conv': 1e-6,
    'bigcn': 1e-6,
    'ebgcn': 1e-6,
    'mode': 'lrp'
}


def show_memory_usage(device):
    t = torch.cuda.get_device_properties(device).total_memory
    r = torch.cuda.memory_reserved(device)
    a = torch.cuda.memory_allocated(device)
    print(f'gpu: {device}\t\ttotal memory: {t}\t\treserved memory: {r}\t\tallocated memory: {a}\t\t'
          f'unallocated memory: {r-a}\t\tfree memory: {t-r}')


def display_mem_usage(device, verbose=False):
    free_mem, total_mem = torch.cuda.mem_get_info(device)
    used_mem = total_mem - free_mem
    mem_used_MB = (total_mem - free_mem) / 1024 ** 2
    mem_proportion_used = used_mem / total_mem
    if verbose:
        print(f'free_mem: {free_mem}\n'
              f'total_mem: {total_mem}\nused_mem: {used_mem}\tmem_used_MB: {mem_used_MB}\n'
              f'mem_proportion_used: {mem_proportion_used}', flush=True)
    return mem_proportion_used


def get_raw_texts_PHEME(root_ids):
    batch_raw_texts = []
    for root_id in root_ids:
        raw_texts = None
        for fold_num in range(9):  # Load raw texts
            try:
                raw_texts, _ = getRawData(FOLD_2_EVENTNAME[fold_num], root_id)
                break
            except:
                pass
        batch_raw_texts.append(raw_texts)
    return batch_raw_texts


def get_model_copy(original_model_type, input_size, hidden_size, output_size, num_class, device, ebgcn_args):
    if original_model_type == 'BiGCN':
        base_model = BiGCN(input_size, hidden_size, output_size, num_class, device).to(device)
    elif original_model_type == 'EBGCN':
        ebgcn_args.input_features = input_size
        ebgcn_args.num_class = num_class
        base_model = EBGCN(ebgcn_args).to(device)
    elif original_model_type == 'CHGAT':
        base_model = CHGAT(input_size, hidden_size, output_size, num_class, device).to(device)
    return base_model


def test_GCN(treeDic, x_test, x_train, TDdroprate, BUdroprate, lr, weight_decay, patience, n_epochs,
              batchsize, datasetname, iter_num, fold, device, **kwargs):
    # version = kwargs.get('version', 2)
    log_file_path = kwargs['log_file_path']
    class_weight = kwargs.get('class_weight', None)
    split_type = kwargs.get('split_type', None)
    model_type = kwargs.get('model_type', None)
    hidden_size = kwargs.get('hidden_size', None)
    output_size = kwargs.get('output_size', None)
    ebgcn_args = kwargs.get('ebgcn_args', None)
    exp_method = kwargs.get('exp_method', None)
    tokeniser = kwargs.get('tokeniser', None)
    text_encoder = kwargs.get('text_encoder', None)
    load_checkpoint = kwargs.get('load_checkpoint', True)
    if class_weight is not None:
        class_weight = class_weight.to(device)
    # print(datasetname, kwargs)
    if datasetname.find('PHEME') != -1 or datasetname == 'NewTwitter':
        input_size = 768
        num_class = 4
    elif datasetname.find('Weibo') != -1:
        input_size = 768
        num_class = 2
    model = get_model_copy(model_type, input_size, hidden_size, output_size, num_class, device,
                                ebgcn_args)
    # Load pretrained model according to model type from path, match iteration and fold
    if load_checkpoint:
        baseline_checkpoints_path = os.path.join(ROOT_DIR, 'testing', f'{datasetname}')
        filenames = os.listdir(baseline_checkpoints_path)
        for criterion in [model_type, f'i{iter_num}', f'f{fold}']:
            filenames = list(filter(lambda x: x.find(f'{criterion}') != -1, filenames))
        savepoint_filename = filenames[0]
        savepoint_name = savepoint_filename[:-3]
        checkpoint_path = os.path.join(baseline_checkpoints_path, savepoint_filename)
        checkpoint_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint_dict['model_state_dict'])
        print(f"model checkpoint loaded from {checkpoint_path}")
        print(f"Epoch: {checkpoint_dict['epoch']}\tTrain Loss: {checkpoint_dict['loss']}\tTrain Acc: {checkpoint_dict['acc']}\n"
              f"Val Loss: {checkpoint_dict['val_loss']}\tVal Acc: {checkpoint_dict['val_acc']}\n"
              f"{checkpoint_dict['res']}\n")
        del checkpoint_dict
    version = f'[{hidden_size},{output_size}]'
    if model_type == 'EBGCN':
        model0 = f'{model_type}-ie'
    else:
        model0 = model_type
    if split_type is not None:
        modelname = f'{model0}-{version}-lr{lr}-wd{weight_decay}-bs{batchsize}-p{patience}'
    expl_save_dir = os.path.join(DATA_DIR, 'explain', f'{datasetname}', f'{model_type}')
    if not os.path.exists(expl_save_dir):
        os.makedirs(expl_save_dir)
    model_ref = copy.deepcopy(model)
    copy_LRP_PARAMS = copy.deepcopy(LRP_PARAMS)
    if exp_method == 'grad-cam':
        LRP_PARAMS['mode'] = 'cam'
    elif exp_method == 'c-eb':
        LRP_PARAMS['mode'] = 'eb'
    else:
        LRP_PARAMS['mode'] = 'lrp'
    try:  # For EBGCN
        model_ref.args.training = False
    except:
        pass
    # print(model_ref, type(model_ref))
    # print(isinstance(model_ref, BiGCN))
    model_copy = lrp_utils.get_lrpwrappermodule(model_ref, LRP_PARAMS).to(device)
    if model_copy is None:
        assert False
    model_copy.eval()
    model.eval()
    if exp_method == 'c-eb':
        copy_LRP_PARAMS['mode'] = 'eb'
        c_eb_model = lrp_utils.get_lrpwrappermodule(model_ref, copy_LRP_PARAMS).to(device)
    traindata_list, testdata_list = loadBiData(datasetname,
                                               treeDic,
                                               x_train,
                                               x_test,
                                               TDdroprate,
                                               BUdroprate)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    data_loader = DataLoader(traindata_list, batch_size=1, shuffle=False, num_workers=5)
    batch_idx = 0
    tqdm_data_loader = tqdm(data_loader)

    def enable_grad_logging(obj):
        """
        function to enable gradient retrieval after backprop from specified Tensor
        @param obj: usually a Tensor
        @return:
        """
        try:
            obj.requires_grad = True
        except:
            pass
        obj.retain_grad()

    def get_embeds_from_token_ids(input_ids, text_encoder, device):
        if type(input_ids) is list:
            input_ids = torch.LongTensor(input_ids).to(device)
        with torch.no_grad():
            # encoded_texts: BatchEncoding = tokeniser(text, padding='longest', max_length=256, truncation=True,
            #                                          return_tensors='pt')
            # if pooling_mode != 'pooler':
            #     embeddings = model.embeddings.word_embeddings(
            #         encoded_texts['input_ids'].to(device)).cpu().detach().numpy()
            #     if pooling_mode == 'mean':
            #         cls = embeddings.mean(-2)
            #         # root_feat = cls[0]
            #     if pooling_mode == 'max':
            #         cls = embeddings.max(-2)
            #         # root_feat = cls[0]
            # elif pooling_mode == 'pooler':
            #     cls = model(encoded_texts['input_ids'].to(device)).pooler_output.cpu().detach().numpy()
                # root_feat = cls[0]
            embeds = text_encoder.embeddings.word_embeddings(input_ids)
        return embeds

    def get_cls_from_token_ids(input_ids, model, device):
        if type(input_ids) is list:
            input_ids = torch.LongTensor(input_ids).to(device).unsqueeze(0)
        model: BertModel
        bert_output = model(input_ids, output_hidden_states=True)
        embeds = bert_output.hidden_states[0].squeeze(0)
        cls = bert_output.pooler_output.squeeze(0)
        return cls, embeds

    def load_and_get_token_embeds(root_ids, model_type, datasetname, text_encoder, device):
        new_x_list, token_embeds_list = [], []
        if datasetname.find('PHEME') != -1:
            token_ids_dir = os.path.join(ROOT_DIR, 'data', 'PHEME', 'raw_text')
        elif datasetname == 'NewTwitter':
            token_ids_dir = os.path.join(ROOT_DIR, 'data', 'NewTwitter')
        elif datasetname == 'NewWeibo':
            token_ids_dir = os.path.join(ROOT_DIR, 'data', 'NewWeibo')
        # elif datatsetname.find('Weibo') != -1:
        #
        # print(Batch_data.x.shape)
        for root_id in root_ids:
            with open(os.path.join(token_ids_dir, f'{root_id}.json'), 'r', encoding='utf-8') as f:
                json_obj = json.load(f)
            token_ids_list = json_obj['token_ids']
            if datasetname.find('PHEME') != -1:
                tokenised_text = json_obj['texts']
            elif datasetname == 'NewTwitter':
                tokenised_text = json_obj['tokenised_texts']
            elif datasetname == 'NewWeibo':
                tokenised_text = json_obj['texts']
            if model_type == 'GACL':
                source_claim = None
            for token_ids in token_ids_list:
                if model_type == 'GACL':
                    prob = torch.ones_like(token_ids) * 0.2
                    prob[0], prob[-1] = 1, 1
                    mask = torch.bernoulli(prob) > 0
                    token_ids = token_ids[mask]
                embeds = get_embeds_from_token_ids(token_ids, text_encoder, device)
                # cls, embeds = get_cls_from_token_ids(token_ids, text_encoder, device)
                embeds.requires_grad = True
                embeds.retain_grad()
                token_embeds_list.append(embeds)
                new_x = embeds.mean(-2)
                # new_x = cls
                if model_type == 'GACL':
                    if source_claim is None:
                        source_claim = new_x
                    root_extend = torch.ones_like(source_claim, dtype=new_x.dtype, device=new_x.device)
                    root_extend *= source_claim
                    new_x = torch.cat((new_x, root_extend), 0)
                new_x_list.append(new_x)
        new_x = torch.stack(new_x_list)
        return new_x, token_embeds_list, (token_ids_list, tokenised_text)

    sample_num = 0
    flipped = [0, 0, 0, 0, 0]
    sample_count = 0
    valid = [0, 0, 0, 0, 0]
    original_eq_all_masked = [0, 0, 0, 0, 0]
    unconstrained_sparsities = []
    flush_count = 0
    time_elapsed = 0
    total_graphs_processed = 0

    for Batch_data, root_ids in tqdm_data_loader:
        try:
            Batch_data.x = Batch_data.cls
        except:
            pass
        Batch_data.to(device)
        flat_idx_2_nested_idx = {}
        nested_idx_2_flat_idx = {}
        test_summary = ''
        graph_expl_dict = {}
        root_id = root_ids[0]
        # print(root_ids, model_type, datasetname, text_encoder)
        if datasetname.find('PHEME') != -1:
            for check_fold in range(9):  # iterate through all source files
                event_name = FOLD_2_EVENTNAME[check_fold]
                with open(os.path.join(DATA_DIR, 'PHEME', f'{event_name}.json'), 'r', encoding='utf-8') as f:
                    json_obj = json.load(f)
                try:
                    root_text = json_obj[f'{root_id}'][f'{root_id}']['text']  # check if entry exists, will throw error if not found
                    break  # forcefully terminate search if found
                except:  # will throw error if no entry, catch and let for loop continue
                    pass
            else:  # if not forcefully termintated, this will execute, skip this iteration in outer loop
                print(f'{root_id} not found')
                continue
        elif datasetname == 'NewTwitter':
            with open(os.path.join(DATA_DIR, 'NewTwitter', f'{root_id}.json'), 'r', encoding='utf-8') as f:
                json_obj = json.load(f)
            root_text = json_obj['tweets'][0]
        elif datasetname == 'NewWeibo':
            with open(os.path.join(DATA_DIR, 'MaWeibo', 'Threads', f'{root_id}.json'), 'r', encoding='utf-8') as f:
                json_obj = json.load(f)
            root_text = json_obj[0]['original_text']
        graph_expl_dict = {'root_id': root_id,
                           'root_text': root_text}
        test_summary += f'{root_id}\t'
        new_x, token_embeds_list, (token_ids_list, tokenised_texts) = load_and_get_token_embeds(
            root_ids, model_type, datasetname, text_encoder=text_encoder, device=device)
        original_claim = Batch_data.root.clone().detach()
        Batch_data.x = new_x.clone().detach()
        Batch_data.root = original_claim.clone().detach()
        num_nodes = new_x.shape[0]
        test_summary += f'{num_nodes}\t'
        # check size of graph
        if num_nodes < 20:  # skip threads with nodes smaller than 20 posts
            sample_num += 1
            continue

        include_root_extend = True
        if exp_method == 'ct-lrp':
            model.eval()
            new_x.retain_grad()
            if include_root_extend:
                original_claim = Batch_data.root
                new_root = Batch_data.root.clone().detach()
                new_root.requires_grad_()
                new_root.retain_grad()
                Batch_data.root = new_root
            # get the flattened index for token and their nested index equivalent
            component_count = 0
            root_components_range = [0, 0]
            for node_num, token_embeds in enumerate(token_embeds_list):
                for token_num in range(token_embeds.shape[0]):
                    flat_idx_2_nested_idx[component_count] = (node_num, token_num)
                    nested_idx_2_flat_idx[(node_num, token_num)] = component_count
                    component_count += 1
            else:  # get root tokens
                if include_root_extend:
                    root_components_range[0] = component_count
                    for token_num in range(token_embeds_list[0].shape[0]):
                        flat_idx_2_nested_idx[component_count] = (-1, token_num)
                        nested_idx_2_flat_idx[(-1, token_num)] = component_count
                        component_count += 1
                    else:
                        root_components_range[1] = component_count
            temp_x = torch.zeros_like(new_x, device=new_x.device)
            Batch_data.x = temp_x
            if include_root_extend:
                temp_claim = torch.zeros_like(original_claim, device=original_claim.device)
                Batch_data.root = temp_claim
            all_masked_probs = model(Batch_data)
            if type(all_masked_probs) is tuple:
                all_masked_probs = all_masked_probs[0]
            _, all_masked_pred = all_masked_probs.max(-1)
            all_masked_logits = model.out.clone().detach()
            Batch_data.x = new_x
            if include_root_extend:
                Batch_data.root = new_root
            original_probs = model(Batch_data)
            if type(original_probs) is tuple:
                original_probs = original_probs[0]
            _, original_pred = original_probs.max(-1)
            original_pred_idx = original_pred.item()
            out_logits = model.out.clone().detach()

            component_scores = torch.zeros((component_count, num_class), device=new_x.device)
            for class_num in range(num_class):
                if Batch_data.x.grad is not None:
                    Batch_data.x.grad.zero_()
                    if include_root_extend:
                        Batch_data.root.grad.zero_()
                with torch.enable_grad():
                    output = model_copy(Batch_data)  # forward
                    if type(output) is tuple:
                        output = output[0]
                output[0, class_num].backward()
                class_x_R = Batch_data.x.grad.detach().clone()
                if include_root_extend:
                    class_claim_R = Batch_data.root.grad.detach().clone()
                token_R_list = []
                for node_num, token_embeds in enumerate(token_embeds_list):
                    temp_list = []
                    node_R = class_x_R[node_num]
                    token_embeds.requires_grad_()
                    token_embeds.retain_grad()
                    if token_embeds.grad is None:
                        pass
                    else:
                        token_embeds.grad.zero_()
                    token_Z = token_embeds.mean(0)
                    token_S = safe_divide(node_R, token_Z, 1e-6, 1e-6)
                    token_Z.backward(token_S)
                    token_R = token_embeds.data * token_embeds.grad
                    token_R_list.append(token_R.clone().detach())
                else:  # compute score for all inputs masked
                    if include_root_extend:
                        token_embeds = token_embeds_list[0]
                        claim_R = class_claim_R
                        token_embeds.requires_grad_()
                        token_embeds.retain_grad()
                        if token_embeds.grad is None:
                            pass
                        else:
                            token_embeds.grad.zero_()
                        token_Z = token_embeds.mean(0, keepdim=True)
                        token_S = safe_divide(claim_R, token_Z, 1e-6, 1e-6)
                        token_Z.backward(token_S)
                        token_R = token_embeds.data * token_embeds.grad
                        claim_R_list = token_R.clone().detach()
                class_scores = torch.cat(token_R_list, dim=0)
                if include_root_extend:
                    class_scores = torch.cat((class_scores, claim_R_list), dim=0)
                component_scores[:, class_num] = class_scores.sum(-1)

            # do the minimsation using ILP
            A_ub, b_ub = [], []  # np.ones((component_count, 3))  # constraints
            A_eq, b_eq = [], []
            # this is the constraint for ILP
            threshold = 0.00
            is_pos = (component_scores > threshold).sum(-1)
            is_shared = is_pos >= 2
            pred_class_scores = component_scores[:, original_pred_idx]
            pos_components = torch.where(
                pred_class_scores > threshold,
                torch.zeros_like(pred_class_scores),
                torch.ones_like(pred_class_scores))
            A_eq.append(pos_components)
            b_eq.append(0)
            is_shared = (pred_class_scores > threshold) & is_shared
            shared_components = torch.where(
                is_shared,
                torch.ones_like(pred_class_scores),
                torch.zeros_like(pred_class_scores))
            # disambiguate shared tokens
            for idx, is_shared_component in enumerate(is_shared):
                if is_shared_component:
                    node_num, token_num = flat_idx_2_nested_idx[idx]
                    if node_num != -1:  # graph nodes
                        temp_x = new_x.clone().detach()
                        temp_token_embeds = token_embeds_list[node_num].clone().detach()
                        temp_token_embeds[token_num] = torch.zeros_like(temp_token_embeds[token_num])
                        temp_x[node_num] = temp_token_embeds.mean(0)
                        Batch_data.x = temp_x
                        Batch_data.root = original_claim
                    else:  # root
                        temp_claim = token_embeds_list[0].clone().detach()
                        temp_claim[token_num] = torch.zeros_like(temp_claim[token_num])
                        temp_claim = temp_claim.mean(0)
                        Batch_data.x = new_x
                        Batch_data.root = temp_claim
                    _ = model(Batch_data)
                    logits = model.out.clone().detach()
                    logits_diff = out_logits - logits
                    if original_pred_idx == logits_diff.argmax():
                        shared_components[idx] = 0
            A_eq.append(shared_components)
            b_eq.append(0)
            A_eq_in = torch.stack(A_eq).cpu().numpy()
            b_eq_in = np.array(b_eq)
            for sparsity_num, sparsity in enumerate([0.0, 0.2, 0.4, 0.6, 0.8]):
                fixed_sparsity = 1 - sparsity
                if sparsity_num == 0:
                    A_ub.append(torch.ones_like(pred_class_scores))
                    max_components = int(component_count * fixed_sparsity)
                    b_ub.append(max_components)
                else:
                    max_components = int(component_count * fixed_sparsity)
                    b_ub[-1] = max_components
                # convert to matrix and vector form
                A_ub_in = torch.stack(A_ub).cpu().numpy()
                b_ub_in = np.array(b_ub)
                c = -pred_class_scores.detach().cpu().numpy()
                new_mask = np.ones_like(c)
                try:
                    solution = linprog(c, A_ub_in, b_ub_in, A_eq_in, b_eq_in, bounds=(0, 1),
                                       method='highs', integrality=np.ones_like(c),
                                       options={'presolve': False})
                    if solution.success:
                        new_mask = solution.x
                except Exception as exce:
                    print(traceback.print_exc(), flush=True)
                    pass
                if sparsity_num == 0:
                    unconstrained_sparsities.append(1 - ((new_mask.sum()) / component_count))
                # generate new masked sample and conduct fidelity test
                masked_x = new_x.clone().detach()
                masked_claim = None
                masked_token_embeds_list = []
                temp_claim_token_embeds = token_embeds_list[0].clone().detach()
                for token_embeds in token_embeds_list:
                    masked_token_embeds_list.append(token_embeds.clone().detach())
                # print(new_mask.shape)
                num_selected_per_node = {}
                for component_num, i in enumerate(new_mask):
                    if i == 1 and component_num != component_count:
                        node_num, token_num = flat_idx_2_nested_idx[component_num]
                        if num_selected_per_node.get(node_num, None) is None:
                            num_selected_per_node[node_num] = 1
                        else:
                            num_selected_per_node[node_num] += 1
                        if node_num != -1:  # mask graph tokens
                            temp_token_embeds = masked_token_embeds_list[node_num]
                            temp_token_embeds[token_num] = torch.zeros_like(
                                temp_token_embeds[token_num],
                                device=temp_token_embeds.device)
                        else:  # mask claim tokens
                            temp_claim_token_embeds[token_num] = torch.zeros_like(
                                temp_claim_token_embeds[token_num],
                                device=temp_claim_token_embeds.device)
                else:
                    if num_selected_per_node.get(-1, None) is not None:
                        num_tokens_removed = num_selected_per_node[-1]
                    else:
                        num_tokens_removed = 0
                    num_tokens = temp_claim_token_embeds.shape[0]
                    masked_claim = temp_claim_token_embeds.sum(0) / (num_tokens - num_tokens_removed)
                for node_num, token_embeds in enumerate(masked_token_embeds_list):
                    if num_selected_per_node.get(node_num, None) is not None:
                        num_tokens_removed = num_selected_per_node[node_num]
                    else:
                        num_tokens_removed = 0
                    num_tokens = token_embeds.shape[0]
                    new_node = token_embeds.sum(0) / (num_tokens - num_tokens_removed)
                    masked_x[node_num] = new_node
                Batch_data.x = masked_x
                if include_root_extend:
                    Batch_data.root = masked_claim
                else:
                    Batch_data.root = masked_x[0]
                fidelity_probs = model(Batch_data)
                if type(fidelity_probs) is tuple:
                    fidelity_probs = fidelity_probs[0]
                _, fidelity_pred = fidelity_probs.max(-1)
                # generate explanation and conduct validity test
                expl_x = torch.zeros_like(new_x, device=new_x.device)
                expl_claim = None
                expl_token_embeds_list = []
                temp_claim_token_embeds = token_embeds_list[0].clone().detach()
                for token_embeds in token_embeds_list:
                    expl_token_embeds_list.append(token_embeds.clone().detach())
                num_selected_per_node = {}
                for component_num, i in enumerate(new_mask):
                    if i != 1 and component_num != component_count:
                        node_num, token_num = flat_idx_2_nested_idx[component_num]
                        if num_selected_per_node.get(node_num, None) is None:
                            num_selected_per_node[node_num] = 1
                        else:
                            num_selected_per_node[node_num] += 1
                        if node_num != -1:  # select explanation graph tokens
                            temp_token_embeds = expl_token_embeds_list[node_num]
                            temp_token_embeds[token_num] = torch.zeros_like(
                                temp_token_embeds[token_num],
                                device=temp_token_embeds.device)
                        else:  # select explanation claim tokens
                            temp_claim_token_embeds[token_num] = torch.zeros_like(
                                temp_claim_token_embeds[token_num],
                                device=temp_claim_token_embeds.device)
                else:
                    if num_selected_per_node.get(-1, None) is not None:
                        num_tokens_selected = num_selected_per_node[-1]
                        expl_claim = temp_claim_token_embeds.sum(0) / num_tokens_selected
                    else:
                        expl_claim = temp_claim_token_embeds.mean(0)
                for node_num, token_embeds in enumerate(expl_token_embeds_list):
                    if num_selected_per_node.get(node_num, None) is not None:
                        num_tokens_selected = num_selected_per_node[node_num]
                        new_node = token_embeds.sum(0) / num_tokens_selected
                        expl_x[node_num] = new_node
                Batch_data.x = expl_x
                if include_root_extend:
                    Batch_data.root = expl_claim
                else:
                    Batch_data.root = expl_x[0]
                validity_probs = model(Batch_data)
                if type(validity_probs) is tuple:
                    validity_probs = validity_probs[0]
                _, validity_pred = validity_probs.max(-1)
                flush = False
                if torch.eq(validity_pred, original_pred):
                    valid[sparsity_num] += 1
                else:
                    flush_count += 1
                    if flush_count >= 50:
                        flush_count = 0
                        flush = True
                    # print(
                    #     f'sample: {sample_num:4d},{sample_count:4d}\tsparsity {sparsity:.1f}\tnot valid\n'
                    #     f'original pred: {original_pred.item()}\tall masked: {all_masked_pred.item()}\t'
                    #     f'validity pred:{validity_pred.item()}', flush=flush)
                if not torch.eq(fidelity_pred, original_pred):
                    flipped[sparsity_num] += 1
                else:
                    flush_count += 1
                    if flush_count >= 50:
                        flush_count = 0
                        flush = True
                    # print(f'sample: {sample_num:4d},{sample_count:4d}\tsparsity {sparsity:.1f}\t'
                    #       f'flip failed', solution.success, solution.status, solution.nit, solution.fun)
                    # print(
                    #     f'original pred: {original_pred.item()}\tall masked: {all_masked_pred.item()}\t'
                    #     f'fidelity pred: {fidelity_pred.item()}', flush=flush)
                    if original_pred.item() == all_masked_pred.item():
                        original_eq_all_masked[sparsity_num] += 1
            sample_count += 1
            sample_num += 1
            continue
        elif exp_method == 'lrp-token':
            model.eval()
            new_x.retain_grad()
            if include_root_extend:
                original_claim = Batch_data.root
                new_root = Batch_data.root.clone().detach()
                new_root.requires_grad_()
                new_root.retain_grad()
                Batch_data.root = new_root
            # get the flattened index for token and their nested index equivalent
            component_count = 0
            root_components_range = [0, 0]
            for node_num, token_embeds in enumerate(token_embeds_list):
                for token_num in range(token_embeds.shape[0]):
                    flat_idx_2_nested_idx[component_count] = (node_num, token_num)
                    nested_idx_2_flat_idx[(node_num, token_num)] = component_count
                    component_count += 1
            else:  # get root tokens
                if include_root_extend:
                    root_components_range[0] = component_count
                    for token_num in range(token_embeds_list[0].shape[0]):
                        flat_idx_2_nested_idx[component_count] = (-1, token_num)
                        nested_idx_2_flat_idx[(-1, token_num)] = component_count
                        component_count += 1
                    else:
                        root_components_range[1] = component_count
            temp_x = torch.zeros_like(new_x, device=new_x.device)
            Batch_data.x = temp_x
            if include_root_extend:
                temp_claim = torch.zeros_like(original_claim, device=original_claim.device)
                Batch_data.root = temp_claim
            all_masked_probs = model(Batch_data)
            if type(all_masked_probs) is tuple:
                all_masked_probs = all_masked_probs[0]
            _, all_masked_pred = all_masked_probs.max(-1)
            all_masked_logits = model.out.clone().detach()
            Batch_data.root = original_claim
            claim_only_probs = model(Batch_data)
            if type(claim_only_probs) is tuple:
                claim_only_probs = claim_only_probs[0]
            _, claim_only_pred = claim_only_probs.max(-1)
            claim_only_logits = model.out.clone().detach()
            Batch_data.x = new_x
            Batch_data.root = torch.zeros_like(original_claim, device=original_claim.device)
            graph_only_probs = model(Batch_data)
            if type(graph_only_probs) is tuple:
                graph_only_probs = graph_only_probs[0]
            _, graph_only_pred = graph_only_probs.max(-1)
            graph_only_logits = model.out.clone().detach()
            Batch_data.x = new_x
            if include_root_extend:
                Batch_data.root = new_root
            original_probs = model(Batch_data)
            if type(original_probs) is tuple:
                original_probs = original_probs[0]
            _, original_pred = original_probs.max(-1)
            original_pred_idx = original_pred.item()
            out_logits = model.out.clone().detach()
            logits_range = out_logits - all_masked_logits
            graph_logits_range = claim_only_logits - all_masked_logits
            claim_logits_range = graph_only_logits - all_masked_logits
            graph_plus_claim_logits_range = graph_logits_range + claim_logits_range

            if Batch_data.x.grad is not None:
                Batch_data.x.grad.zero_()
                if include_root_extend:
                    Batch_data.root.grad.zero_()
            with torch.enable_grad():
                output = model_copy(Batch_data)  # forward
                if type(output) is tuple:
                    output = output[0]
            output[0, original_pred_idx].backward()
            class_x_R = Batch_data.x.grad.detach().clone()
            if include_root_extend:
                class_claim_R = Batch_data.root.grad.detach().clone()
            token_R_list = []
            for node_num, token_embeds in enumerate(token_embeds_list):
                temp_list = []
                node_R = class_x_R[node_num]
                token_embeds.requires_grad_()
                token_embeds.retain_grad()
                if token_embeds.grad is None:
                    pass
                else:
                    token_embeds.grad.zero_()
                token_Z = token_embeds.mean(0)
                token_S = safe_divide(node_R, token_Z, 1e-6, 1e-6)
                token_Z.backward(token_S)
                token_R = token_embeds.data * token_embeds.grad
                token_R_list.append(token_R.clone().detach())
            else:  # compute score for all inputs masked
                if include_root_extend:
                    token_embeds = token_embeds_list[0]
                    claim_R = class_claim_R
                    token_embeds.requires_grad_()
                    token_embeds.retain_grad()
                    if token_embeds.grad is None:
                        pass
                    else:
                        token_embeds.grad.zero_()
                    token_Z = token_embeds.mean(0, keepdim=True)
                    token_S = safe_divide(claim_R, token_Z, 1e-6, 1e-6)
                    token_Z.backward(token_S)
                    token_R = token_embeds.data * token_embeds.grad
                    claim_R_list = token_R.clone().detach()
            component_scores = torch.cat(token_R_list, dim=0)
            if include_root_extend:
                component_scores = torch.cat((component_scores, claim_R_list), dim=0)
            component_scores = component_scores.sum(-1)
            # do the minimsation using ILP
            A_ub, b_ub = [], []  # np.ones((component_count, 3))  # constraints
            A_eq, b_eq = [], []
            # this is the constraint for ILP
            threshold = 0.00
            pos_components = torch.where(
                component_scores > threshold,
                torch.zeros_like(component_scores),
                torch.ones_like(component_scores))
            A_eq.append(pos_components)
            b_eq.append(0)
            A_eq_in = torch.stack(A_eq).cpu().numpy()
            b_eq_in = np.array(b_eq)
            for sparsity_num, sparsity in enumerate([0.0, 0.2, 0.4, 0.6, 0.8]):
                fixed_sparsity = 1 - sparsity
                if sparsity_num == 0:
                    A_ub.append(torch.ones_like(component_scores))
                    max_components = int(component_count * fixed_sparsity)
                    b_ub.append(max_components)
                else:
                    max_components = int(component_count * fixed_sparsity)
                    b_ub[-1] = max_components
                # convert to matrix and vector form
                A_ub_in = torch.stack(A_ub).cpu().numpy()
                b_ub_in = np.array(b_ub)
                solution_intersect = None
                c = -component_scores.detach().cpu().numpy()
                new_mask = np.ones_like(c)
                try:
                    solution = linprog(c, A_ub_in, b_ub_in, A_eq_in, b_eq_in, bounds=(0, 1),
                                       method='highs', integrality=np.ones_like(c),
                                       options={'presolve': False})
                    if solution.success:
                        new_mask = solution.x
                except Exception as exce:
                    print(traceback.print_exc(), flush=True)
                    pass
                if sparsity_num == 0:
                    unconstrained_sparsities.append(1 - ((new_mask.sum()) / component_count))
                # generate new masked sample and conduct fidelity test
                masked_x = new_x.clone().detach()
                masked_claim = None
                masked_token_embeds_list = []
                temp_claim_token_embeds = token_embeds_list[0].clone().detach()
                for token_embeds in token_embeds_list:
                    masked_token_embeds_list.append(token_embeds.clone().detach())
                num_selected_per_node = {}
                for component_num, i in enumerate(new_mask):
                    if i == 1 and component_num != component_count:
                        node_num, token_num = flat_idx_2_nested_idx[component_num]
                        if num_selected_per_node.get(node_num, None) is None:
                            num_selected_per_node[node_num] = 1
                        else:
                            num_selected_per_node[node_num] += 1
                        if node_num != -1:  # mask graph tokens
                            temp_token_embeds = masked_token_embeds_list[node_num]
                            temp_token_embeds[token_num] = torch.zeros_like(
                                temp_token_embeds[token_num],
                                device=temp_token_embeds.device)
                        else:  # mask claim tokens
                            temp_claim_token_embeds[token_num] = torch.zeros_like(
                                temp_claim_token_embeds[token_num],
                                device=temp_token_embeds.device)
                else:
                    if num_selected_per_node.get(-1, None) is not None:
                        num_tokens_removed = num_selected_per_node[-1]
                    else:
                        num_tokens_removed = 0
                    num_tokens = temp_claim_token_embeds.shape[0]
                    masked_claim = temp_claim_token_embeds.sum(0) / (num_tokens - num_tokens_removed)
                for node_num, token_embeds in enumerate(masked_token_embeds_list):
                    if num_selected_per_node.get(node_num, None) is not None:
                        num_tokens_removed = num_selected_per_node[node_num]
                    else:
                        num_tokens_removed = 0
                    num_tokens = token_embeds.shape[0]
                    new_node = token_embeds.sum(0) / (num_tokens - num_tokens_removed)
                    masked_x[node_num] = new_node
                Batch_data.x = masked_x
                if include_root_extend:
                    Batch_data.root = masked_claim
                else:
                    Batch_data.root = masked_x[0]
                fidelity_probs = model(Batch_data)
                if type(fidelity_probs) is tuple:
                    fidelity_probs = fidelity_probs[0]
                _, fidelity_pred = fidelity_probs.max(-1)
                # generate explanation and conduct validity test
                # expl_x = new_x.clone().detach()
                expl_x = torch.zeros_like(new_x, device=new_x.device)
                expl_claim = None
                expl_token_embeds_list = []
                temp_claim_token_embeds = token_embeds_list[0].clone().detach()
                for token_embeds in token_embeds_list:
                    expl_token_embeds_list.append(token_embeds.clone().detach())
                num_selected_per_node = {}
                for component_num, i in enumerate(new_mask):
                    if i != 1 and component_num != component_count:
                        node_num, token_num = flat_idx_2_nested_idx[component_num]
                        if num_selected_per_node.get(node_num, None) is None:
                            num_selected_per_node[node_num] = 1
                        else:
                            num_selected_per_node[node_num] += 1
                        if node_num != -1:  # select explanation graph tokens
                            temp_token_embeds = expl_token_embeds_list[node_num]
                            temp_token_embeds[token_num] = torch.zeros_like(
                                temp_token_embeds[token_num],
                                device=temp_token_embeds.device)
                        else:  # select explanation claim tokens
                            temp_claim_token_embeds[token_num] = torch.zeros_like(
                                temp_claim_token_embeds[token_num],
                                device=temp_token_embeds.device)
                else:
                    if num_selected_per_node.get(-1, None) is not None:
                        num_tokens_selected = num_selected_per_node[-1]
                        expl_claim = temp_claim_token_embeds.sum(0) / num_tokens_selected
                    else:
                        expl_claim = temp_claim_token_embeds.mean(0)
                for node_num, token_embeds in enumerate(expl_token_embeds_list):
                    if num_selected_per_node.get(node_num, None) is not None:
                        num_tokens_selected = num_selected_per_node[node_num]
                        new_node = token_embeds.sum(0) / num_tokens_selected
                        expl_x[node_num] = new_node
                Batch_data.x = expl_x
                if include_root_extend:
                    Batch_data.root = expl_claim
                else:
                    Batch_data.root = expl_x[0]
                validity_probs = model(Batch_data)
                if type(validity_probs) is tuple:
                    validity_probs = validity_probs[0]
                _, validity_pred = validity_probs.max(-1)
                flush = False
                if torch.eq(validity_pred, original_pred):
                    valid[sparsity_num] += 1
                else:
                    flush_count += 1
                    if flush_count >= 50:
                        flush_count = 0
                        flush = True
                    # print(
                    #     f'sample: {sample_num:4d},{sample_count:4d}\tsparsity {sparsity:.1f}\tnot valid\n'
                    #     f'original pred: {original_pred.item()}\tall masked: {all_masked_pred.item()}\t'
                    #     f'validity pred:{validity_pred.item()}', flush=flush)
                if not torch.eq(fidelity_pred, original_pred):
                    flipped[sparsity_num] += 1
                else:
                    flush_count += 1
                    if flush_count >= 50:
                        flush_count = 0
                        flush = True
                    # print(f'sample: {sample_num:4d},{sample_count:4d}\tsparsity {sparsity:.1f}\t'
                    #       f'flip failed', solution.success, solution.status, solution.nit, solution.fun)
                    # print(
                    #     f'original pred: {original_pred.item()}\tall masked: {all_masked_pred.item()}\t'
                    #     f'fidelity pred: {fidelity_pred.item()}', flush=flush)
                    if original_pred.item() == all_masked_pred.item():
                        original_eq_all_masked[sparsity_num] += 1
            sample_count += 1
            sample_num += 1
            continue
        elif exp_method == 'lrp':
            model.eval()
            new_x.retain_grad()
            original_claim = Batch_data.root
            new_root = Batch_data.root.clone().detach()
            new_root.requires_grad_()
            new_root.retain_grad()
            Batch_data.root = new_root
            # get the flattened index for token and their nested index equivalent
            component_count = new_x.shape[0]
            temp_x = torch.zeros_like(new_x, device=new_x.device)
            Batch_data.x = temp_x
            if include_root_extend:
                temp_claim = torch.zeros_like(original_claim, device=original_claim.device)
                Batch_data.root = temp_claim
            all_masked_probs = model(Batch_data)
            if type(all_masked_probs) is tuple:
                all_masked_probs = all_masked_probs[0]
            _, all_masked_pred = all_masked_probs.max(-1)
            all_masked_logits = model.out.clone().detach()
            Batch_data.root = original_claim
            claim_only_probs = model(Batch_data)
            if type(claim_only_probs) is tuple:
                claim_only_probs = claim_only_probs[0]
            _, claim_only_pred = claim_only_probs.max(-1)
            claim_only_logits = model.out.clone().detach()
            Batch_data.x = new_x
            Batch_data.root = torch.zeros_like(original_claim, device=original_claim.device)
            graph_only_probs = model(Batch_data)
            if type(graph_only_probs) is tuple:
                graph_only_probs = graph_only_probs[0]
            _, graph_only_pred = graph_only_probs.max(-1)
            graph_only_logits = model.out.clone().detach()
            Batch_data.x = new_x
            if include_root_extend:
                Batch_data.root = new_root
            original_probs = model(Batch_data)
            if type(original_probs) is tuple:
                original_probs = original_probs[0]
            _, original_pred = original_probs.max(-1)
            original_pred_idx = original_pred.item()
            out_logits = model.out.clone().detach()
            logits_range = out_logits - all_masked_logits
            graph_logits_range = claim_only_logits - all_masked_logits
            claim_logits_range = graph_only_logits - all_masked_logits
            graph_plus_claim_logits_range = graph_logits_range + claim_logits_range

            if Batch_data.x.grad is not None:
                Batch_data.x.grad.zero_()
                if include_root_extend:
                    Batch_data.root.grad.zero_()
            with torch.enable_grad():
                output = model_copy(Batch_data)  # forward
                if type(output) is tuple:
                    output = output[0]
            output[0, original_pred_idx].backward()
            class_x_R = Batch_data.x.grad.detach().clone()
            component_scores = class_x_R.sum(-1)
            # do the minimsation using ILP
            A_ub, b_ub = [], []  # np.ones((component_count, 3))  # constraints
            A_eq, b_eq = [], []
            # this is the constraint for ILP
            threshold = 0.00
            pos_components = torch.where(
                component_scores > threshold,
                torch.zeros_like(component_scores),
                torch.ones_like(component_scores))
            A_eq.append(pos_components)
            b_eq.append(0)
            A_eq_in = torch.stack(A_eq).cpu().numpy()
            b_eq_in = np.array(b_eq)
            for sparsity_num, sparsity in enumerate([0.0, 0.2, 0.4, 0.6, 0.8]):
                fixed_sparsity = 1 - sparsity
                if sparsity_num == 0:
                    A_ub.append(torch.ones_like(component_scores))
                    max_components = int(component_count * fixed_sparsity)
                    b_ub.append(max_components)
                else:
                    max_components = int(component_count * fixed_sparsity)
                    b_ub[-1] = max_components
                # convert to matrix and vector form
                A_ub_in = torch.stack(A_ub).cpu().numpy()
                b_ub_in = np.array(b_ub)
                c = -component_scores.detach().cpu().numpy()
                new_mask = np.ones_like(c)
                try:
                    solution = linprog(c, A_ub_in, b_ub_in, A_eq_in, b_eq_in, bounds=(0, 1),
                                       method='highs', integrality=np.ones_like(c),
                                       options={'presolve': False})
                    if solution.success:
                        new_mask = solution.x
                except Exception as exce:
                    # print(traceback.print_exc(), flush=True)
                    pass
                if sparsity_num == 0:
                    unconstrained_sparsities.append(1 - ((new_mask.sum()) / component_count))
                # generate new masked sample and conduct fidelity test
                masked_x = new_x.clone().detach()
                masked_claim = original_claim.clone().detach()
                for node_num, i in enumerate(new_mask):
                    if i == 1:
                        masked_x[node_num] = torch.zeros_like(masked_x[node_num], device=masked_x.device)
                        if node_num == 0:
                            masked_claim = torch.zeros_like(masked_claim, device=masked_x.device)
                else:
                    Batch_data.x = masked_x
                    Batch_data.root = masked_claim
                fidelity_probs = model(Batch_data)
                if type(fidelity_probs) is tuple:
                    fidelity_probs = fidelity_probs[0]
                _, fidelity_pred = fidelity_probs.max(-1)
                # generate explanation and conduct validity test
                expl_x = torch.zeros_like(new_x, device=new_x.device)
                expl_claim = torch.zeros_like(original_claim, device=new_x.device)
                for node_num, i in enumerate(new_mask):
                    if i != 1:
                        expl_x[node_num] = new_x[node_num].clone().detach()
                        if node_num == 0:
                            expl_claim = original_claim.clone().detach()
                else:
                    Batch_data.x = expl_x
                    Batch_data.root = expl_claim
                validity_probs = model(Batch_data)
                if type(validity_probs) is tuple:
                    validity_probs = validity_probs[0]
                _, validity_pred = validity_probs.max(-1)
                flush = False
                if torch.eq(validity_pred, original_pred):
                    valid[sparsity_num] += 1
                else:
                    flush_count += 1
                    if flush_count >= 50:
                        flush_count = 0
                        flush = True
                    # print(
                    #     f'sample: {sample_num:4d},{sample_count:4d}\tsparsity {sparsity:.1f}\tnot valid\n'
                    #     f'original pred: {original_pred.item()}\tall masked: {all_masked_pred.item()}\t'
                    #     f'validity pred:{validity_pred.item()}', flush=flush)
                if not torch.eq(fidelity_pred, original_pred):
                    flipped[sparsity_num] += 1
                else:
                    flush_count += 1
                    if flush_count >= 50:
                        flush_count = 0
                        flush = True
                    # print(f'sample: {sample_num:4d},{sample_count:4d}\tsparsity {sparsity:.1f}\t'
                    #       f'flip failed', solution.success, solution.status, solution.nit, solution.fun)
                    # print(
                    #     f'original pred: {original_pred.item()}\tall masked: {all_masked_pred.item()}\t'
                    #     f'fidelity pred: {fidelity_pred.item()}', flush=flush)
                    if original_pred.item() == all_masked_pred.item():
                        original_eq_all_masked[sparsity_num] += 1
            sample_count += 1
            sample_num += 1
            continue
        elif exp_method == 'c-eb':
            c_eb_model.eval()
            new_x.retain_grad()
            original_claim = Batch_data.root
            new_root = Batch_data.root.clone().detach()
            new_root.requires_grad_()
            new_root.retain_grad()
            Batch_data.root = new_root
            component_count = new_x.shape[0]
            temp_x = torch.zeros_like(new_x, device=new_x.device)
            Batch_data.x = temp_x
            if include_root_extend:
                temp_claim = torch.zeros_like(original_claim, device=original_claim.device)
                Batch_data.root = temp_claim
            all_masked_probs = model(Batch_data)
            if type(all_masked_probs) is tuple:
                all_masked_probs = all_masked_probs[0]
            _, all_masked_pred = all_masked_probs.max(-1)
            Batch_data.x = new_x
            if include_root_extend:
                Batch_data.root = new_root
            original_probs = model(Batch_data)
            if type(original_probs) is tuple:
                original_probs = original_probs[0]
            _, original_pred = original_probs.max(-1)
            original_pred_idx = original_pred.item()
            if Batch_data.x.grad is not None:
                Batch_data.x.grad.zero_()
                if include_root_extend:
                    Batch_data.root.grad.zero_()
            with torch.enable_grad():
                output = c_eb_model(Batch_data)  # forward
                if type(output) is tuple:
                    output = output[0]
            output[0, original_pred_idx].backward()
            class_x_R = Batch_data.x.grad.detach().clone()
            component_scores = class_x_R.sum(-1)
            # do the minimsation using ILP
            A_ub, b_ub = [], []  # np.ones((component_count, 3))  # constraints
            A_eq, b_eq = [], []
            # this is the constraint for ILP
            threshold = 0.00
            pos_components = torch.where(
                component_scores > threshold,
                torch.zeros_like(component_scores),
                torch.ones_like(component_scores))
            A_eq.append(pos_components)
            b_eq.append(0)
            A_eq_in = torch.stack(A_eq).cpu().numpy()
            b_eq_in = np.array(b_eq)
            for sparsity_num, sparsity in enumerate([0.0, 0.2, 0.4, 0.6, 0.8]):
                fixed_sparsity = 1 - sparsity
                if sparsity_num == 0:
                    A_ub.append(torch.ones_like(component_scores))
                    max_components = int(component_count * fixed_sparsity)
                    b_ub.append(max_components)
                else:
                    max_components = int(component_count * fixed_sparsity)
                    b_ub[-1] = max_components
                # convert to matrix and vector form
                A_ub_in = torch.stack(A_ub).cpu().numpy()
                b_ub_in = np.array(b_ub)
                c = -component_scores.detach().cpu().numpy()
                new_mask = np.ones_like(c)
                try:
                    solution = linprog(c, A_ub_in, b_ub_in, A_eq_in, b_eq_in, bounds=(0, 1),
                                       method='highs', integrality=np.ones_like(c),
                                       options={'presolve': False})
                    if solution.success:
                        new_mask = solution.x
                except Exception as exce:
                    # print(traceback.print_exc(), flush=True)
                    pass
                if sparsity_num == 0:
                    unconstrained_sparsities.append(1 - ((new_mask.sum()) / component_count))
                # generate new masked sample and conduct fidelity test
                masked_x = new_x.clone().detach()
                masked_claim = original_claim.clone().detach()
                for node_num, i in enumerate(new_mask):
                    if i == 1:
                        masked_x[node_num] = torch.zeros_like(masked_x[node_num], device=masked_x.device)
                        if node_num == 0:
                            masked_claim = torch.zeros_like(masked_claim, device=masked_x.device)
                else:
                    Batch_data.x = masked_x
                    Batch_data.root = masked_claim
                fidelity_probs = model(Batch_data)
                if type(fidelity_probs) is tuple:
                    fidelity_probs = fidelity_probs[0]
                _, fidelity_pred = fidelity_probs.max(-1)
                # generate explanation and conduct validity test
                expl_x = torch.zeros_like(new_x, device=new_x.device)
                expl_claim = torch.zeros_like(original_claim, device=new_x.device)
                for node_num, i in enumerate(new_mask):
                    if i != 1:
                        expl_x[node_num] = new_x[node_num].clone().detach()
                        if node_num == 0:
                            expl_claim = original_claim.clone().detach()
                else:
                    Batch_data.x = expl_x
                    Batch_data.root = expl_claim
                validity_probs = model(Batch_data)
                if type(validity_probs) is tuple:
                    validity_probs = validity_probs[0]
                _, validity_pred = validity_probs.max(-1)
                flush = False
                if torch.eq(validity_pred, original_pred):
                    valid[sparsity_num] += 1
                else:
                    flush_count += 1
                    if flush_count >= 50:
                        flush_count = 0
                        flush = True
                    # print(
                    #     f'sample: {sample_num:4d},{sample_count:4d}\tsparsity {sparsity:.1f}\tnot valid\n'
                    #     f'original pred: {original_pred.item()}\tall masked: {all_masked_pred.item()}\t'
                    #     f'validity pred:{validity_pred.item()}', flush=flush)
                if not torch.eq(fidelity_pred, original_pred):
                    flipped[sparsity_num] += 1
                else:
                    flush_count += 1
                    if flush_count >= 50:
                        flush_count = 0
                        flush = True
                    # print(f'sample: {sample_num:4d},{sample_count:4d}\tsparsity {sparsity:.1f}\t'
                    #       f'flip failed', solution.success, solution.status, solution.nit, solution.fun)
                    # print(
                    #     f'original pred: {original_pred.item()}\tall masked: {all_masked_pred.item()}\t'
                    #     f'fidelity pred: {fidelity_pred.item()}', flush=flush)
                    if original_pred.item() == all_masked_pred.item():
                        original_eq_all_masked[sparsity_num] += 1
            sample_count += 1
            sample_num += 1
            continue
        elif exp_method == 'grad-cam':
            temp_x = torch.zeros_like(new_x, device=new_x.device)
            Batch_data.x = temp_x
            temp_claim = torch.zeros_like(original_claim, device=original_claim.device)
            Batch_data.root = temp_claim
            all_masked_probs = model(Batch_data)
            if type(all_masked_probs) is tuple:
                all_masked_probs = all_masked_probs[0]
            _, all_masked_pred = all_masked_probs.max(-1)

            Batch_data.x = new_x
            new_x.retain_grad()
            Batch_data.root = original_claim.clone().detach()
            with torch.enable_grad():
                if model_type != 'EBGCN':
                    output = model(Batch_data)
                else:
                    output, _, _ = model(Batch_data)
            _, pred = output.max(dim=-1)
            reference_output_value = output[0, pred]
            original_pred = pred
            node_embeds = model.get_node_embeds_after_forward(layer=1)
            if type(node_embeds) is tuple:  # for directed graph
                node_embeds[0].retain_grad()
                node_embeds[1].retain_grad()
            else:
                node_embeds.retain_grad()
            output[0, pred].backward()
            if type(node_embeds) is tuple:
                alpha0 = node_embeds[0].grad.mean(0)
                alpha1 = node_embeds[1].grad.mean(0)
                node_grads = torch.cat((alpha0 * node_embeds[0], alpha1 * node_embeds[1]), dim=-1)
            else:
                alpha = node_embeds.grad.mean(0)
                node_grads = alpha * node_embeds
            component_scores = node_grads.sum(-1)
            component_count = component_scores.shape[0]
            # do the minimsation using ILP
            A_ub, b_ub = [], []  # np.ones((component_count, 3))  # constraints
            A_eq, b_eq = [], []
            # this is the constraint for ILP
            threshold = 0.00
            pos_components = torch.where(
                component_scores > threshold,
                torch.zeros_like(component_scores),
                torch.ones_like(component_scores))
            A_eq.append(pos_components)
            b_eq.append(0)
            A_eq_in = torch.stack(A_eq).cpu().numpy()
            b_eq_in = np.array(b_eq)
            for sparsity_num, sparsity in enumerate([0.0, 0.2, 0.4, 0.6, 0.8]):
                fixed_sparsity = 1 - sparsity
                if sparsity_num == 0:
                    A_ub.append(torch.ones_like(component_scores))
                    max_components = int(component_count * fixed_sparsity)
                    b_ub.append(max_components)
                else:
                    max_components = int(component_count * fixed_sparsity)
                    b_ub[-1] = max_components
                # convert to matrix and vector form
                A_ub_in = torch.stack(A_ub).cpu().numpy()
                b_ub_in = np.array(b_ub)
                c = -component_scores.detach().cpu().numpy()
                new_mask = np.ones_like(c)
                try:
                    solution = linprog(c, A_ub_in, b_ub_in, A_eq_in, b_eq_in, bounds=(0, 1),
                                       method='highs', integrality=np.ones_like(c),
                                       options={'presolve': False})
                    if solution.success:
                        new_mask = solution.x
                except Exception as exce:
                    print(traceback.print_exc(), flush=True)
                    pass
                if sparsity_num == 0:
                    unconstrained_sparsities.append(1 - ((new_mask.sum()) / component_count))
                # generate new masked sample and conduct fidelity test
                masked_x = new_x.clone().detach()
                masked_claim = original_claim.clone().detach()
                for node_num, i in enumerate(new_mask):
                    if i == 1:
                        masked_x[node_num] = torch.zeros_like(masked_x[node_num], device=masked_x.device)
                        if node_num == 0:
                            masked_claim = torch.zeros_like(masked_claim, device=masked_x.device)
                else:
                    Batch_data.x = masked_x
                    Batch_data.root = masked_claim
                fidelity_probs = model(Batch_data)
                if type(fidelity_probs) is tuple:
                    fidelity_probs = fidelity_probs[0]
                _, fidelity_pred = fidelity_probs.max(-1)
                # generate explanation and conduct validity test
                expl_x = torch.zeros_like(new_x, device=new_x.device)
                expl_claim = torch.zeros_like(original_claim, device=new_x.device)
                for node_num, i in enumerate(new_mask):
                    if i != 1:
                        expl_x[node_num] = new_x[node_num].clone().detach()
                        if node_num == 0:
                            expl_claim = original_claim.clone().detach()
                else:
                    Batch_data.x = expl_x
                    Batch_data.root = expl_claim
                validity_probs = model(Batch_data)
                if type(validity_probs) is tuple:
                    validity_probs = validity_probs[0]
                _, validity_pred = validity_probs.max(-1)
                flush = False
                if torch.eq(validity_pred, original_pred):
                    valid[sparsity_num] += 1
                else:
                    flush_count += 1
                    if flush_count >= 50:
                        flush_count = 0
                        flush = True
                    # print(
                    #     f'sample: {sample_num:4d},{sample_count:4d}\tsparsity {sparsity:.1f}\tnot valid\n'
                    #     f'original pred: {original_pred.item()}\tall masked: {all_masked_pred.item()}\t'
                    #     f'validity pred:{validity_pred.item()}', flush=flush)
                if not torch.eq(fidelity_pred, original_pred):
                    flipped[sparsity_num] += 1
                else:
                    flush_count += 1
                    if flush_count >= 50:
                        flush_count = 0
                        flush = True
                    # print(f'sample: {sample_num:4d},{sample_count:4d}\tsparsity {sparsity:.1f}\t'
                    #       f'flip failed', solution.success, solution.status, solution.nit, solution.fun)
                    # print(
                    #     f'original pred: {original_pred.item()}\tall masked: {all_masked_pred.item()}\t'
                    #     f'fidelity pred: {fidelity_pred.item()}', flush=flush)
                    if original_pred.item() == all_masked_pred.item():
                        original_eq_all_masked[sparsity_num] += 1
            sample_count += 1
            sample_num += 1
            continue
    fold_summary = ''
    print(f'fold {fold}\tavg sparsity (unconstrained): {np.array(unconstrained_sparsities).mean():.4f}')
    fold_summary += f'fold {fold}\tavg sparsity (unconstrained): {np.array(unconstrained_sparsities).mean():.4f}\n'
    for sparsity_num, sparsity in enumerate([0.0, 0.2, 0.4, 0.6, 0.8]):
        print(f'sparsity: {sparsity}')
        print(f'validity: {valid[sparsity_num]/sample_count:.4f} [{valid[sparsity_num]}/{sample_count}]')
        print(f'fidelity: {flipped[sparsity_num]/sample_count:.4f} [{flipped[sparsity_num]}/{sample_count}]')
        no_flip_count = sample_count - flipped[sparsity_num]
        if no_flip_count != 0:
            print(f'original == all masked: {original_eq_all_masked[sparsity_num]/no_flip_count:.4f} '
                  f'[{original_eq_all_masked[sparsity_num]}/{no_flip_count}]')
        fold_summary += f'sparsity: {sparsity}\n' \
                        f'validity: {valid[sparsity_num]/sample_count:.4f} [{valid[sparsity_num]}/{sample_count}]\n' \
                        f'fidelity: {flipped[sparsity_num]/sample_count:.4f} [{flipped[sparsity_num]}/{sample_count}]\n'
        if no_flip_count != 0:
            fold_summary += f'original == all masked: {original_eq_all_masked[sparsity_num]/no_flip_count:.4f} ' \
                            f'[{original_eq_all_masked[sparsity_num]}/{no_flip_count}]\n'
    with open(log_file_path, 'a',) as f:
        f.write(fold_summary)
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--datasetname', type=str, default="Twitter", metavar='dataname',
                        help='dataset name, option: Twitter/PHEME/Weibo', choices=['Twitter', 'PHEME', 'Weibo'])
    parser.add_argument('-m', '--modelname', type=str, default="BiGCN", metavar='modeltype',
                        help='model type, option: BiGCN/EBGCN/CHGAT', choices=['BiGCN', 'EBGCN', 'CHGAT'])
    parser.add_argument('--input_features', type=int, default=768, metavar='inputF',
                        help='dimension of input features (BERT)')
    parser.add_argument('--hidden_features', type=int, default=64, metavar='graph_hidden',
                        help='dimension of graph hidden state')
    parser.add_argument('--output_features', type=int, default=64, metavar='output_features',
                        help='dimension of output features')
    parser.add_argument('--num_class', type=int, default=4, metavar='numclass',
                        help='number of classes')
    parser.add_argument('--num_workers', type=int, default=0, metavar='num_workers',
                        help='number of workers for training')

    # Parameters for training the model
    parser.add_argument('--seed', type=int, default=2020, help='random state seed')
    parser.add_argument('--no_cuda', action='store_true',
                        help='does not use GPU')
    parser.add_argument('--num_cuda', type=int, default=0,
                        help='index of GPU 0/1')

    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate')
    parser.add_argument('--lr_scale_bu', type=int, default=5, metavar='LRSB',
                        help='learning rate scale for bottom-up direction')
    parser.add_argument('--lr_scale_td', type=int, default=1, metavar='LRST',
                        help='learning rate scale for top-down direction')
    parser.add_argument('--l2', type=float, default=1e-4, metavar='L2',
                        help='L2 regularization weight')

    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout',
                        help='dropout rate')
    parser.add_argument('--patience', type=int, default=10, metavar='patience',
                        help='patience for early stop')
    parser.add_argument('--batchsize', type=int, default=128, metavar='BS',
                        help='batch size')
    parser.add_argument('--n_epochs', type=int, default=200, metavar='E',
                        help='number of max epochs')
    parser.add_argument('--iterations', type=int, default=1, metavar='F',
                        help='number of iterations for 5-fold cross-validation')

    # Parameters for the proposed model
    parser.add_argument('--TDdroprate', type=float, default=0, metavar='TDdroprate',
                        help='drop rate for edges in the top-down propagation graph')
    parser.add_argument('--BUdroprate', type=float, default=0, metavar='BUdroprate',
                        help='drop rate for edges in the bottom-up dispersion graph')
    parser.add_argument('--edge_infer_td', action='store_true', default=True,  # default=False,
                        help='edge inference in the top-down graph')
    parser.add_argument('--edge_infer_bu', action='store_true', default=True,  # default=True,
                        help='edge inference in the bottom-up graph')
    parser.add_argument('--edge_loss_td', type=float, default=0.2, metavar='edge_loss_td',
                        help='a hyperparameter gamma to weight the unsupervised relation learning loss in the top-down propagation graph')
    parser.add_argument('--edge_loss_bu', type=float, default=0.2, metavar='edge_loss_bu',
                        help='a hyperparameter gamma to weight the unsupervised relation learning loss in the bottom-up dispersion graph')
    parser.add_argument('--edge_num', type=int, default=2, metavar='edgenum',
                        help='latent relation types T in the edge inference')

    parser.add_argument('--exp_method', type=str, default='lrp', metavar='exp_method',
                        help='explanation method, option: ct-lrp/lrp-token/lrp/grad-cam/c-eb',
                        choices=['ct-lrp', 'lrp-token', 'lrp', 'grad-cam', 'c-eb'])

    args = parser.parse_args()

    # some admin stuff
    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.num_cuda}' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    datasetname = f'New{args.datasetname}'  # 'NewTwitter', 'NewPHEME', 'NewWeibo'
    # iterations = int(sys.argv[2])
    args.datasetname = datasetname
    args.input_features = 768
    args.device = device
    args.training = True

    model = args.modelname
    model = 'EBGCN'  # 'BiGCN', 'EBGCN', 'CHGAT'
    treeDic = None  # Not required for PHEME
    min_graph_size = 20  # Exclude graphs with less that this number of nodes

    lr = args.lr  # 5e-4
    weight_decay = args.l2  # 1e-4
    patience = args.patience  # 10
    n_epochs = args.n_epochs  # 200
    batchsize = args.batchsize
    iterations = args.iterations
    hidden_size = args.hidden_features
    output_size = args.output_features
    TDdroprate = args.TDdroprate
    BUdroprate = args.BUdroprate
    # edge_dropout = 0.2  # 0.2
    # exp_method = args.exp_method
    exp_method = 'grad-cam'  # 'ct-lrp', 'lrp-token', 'lrp', 'grad-cam', 'c-eb'

    SAVE_DIR_PATH = os.path.join(EXPLAIN_DIR, datasetname, 'temp', exp_method)
    if not os.path.exists(SAVE_DIR_PATH):
        os.makedirs(SAVE_DIR_PATH)

    print(device)
    if datasetname in ['NewTwitter', 'NewWeibo', 'NewPHEME']:
        # bert_tokeniser = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
        # bert_tokeniser = BertTokenizer(torch.load("tokenizer_config.json"))
        # bert_tokeniser.load_state_dict(torch.load("tokenizer.json"))
        # bert_model = BertModel.from_pretrained('bert-base-multilingual-uncased').to(device)
        bert_tokeniser = BertTokenizer.from_pretrained('./bert-dir')
        bert_model = BertModel.from_pretrained('./bert-dir').to(device)
        # torch.save(bert_model.config, "temp-bert-config.pt")
        # torch.save(bert_model.state_dict(), "temp-bert.pt")
        bert_model.load_state_dict(torch.load("temp-bert.pt", map_location='cpu'))
        bert_model.eval()
        print(next(bert_model.parameters()).device)
    else:
        bert_tokeniser, bert_model = None, None

    for datasetname in ['NewTwitter', 'NewWeibo', 'NewPHEME']:
        args.datasetname = datasetname
        version = f'[{hidden_size},{output_size}]'
        split_type = '5fold' if datasetname.find('PHEME') != -1 else '9fold'  # '5fold', '9fold'
        log_file_path = f'{datasetname}_log.txt'
        if model == 'EBGCN':
            model0 = f'{model}-ie'
        else:
            model0 = model
        log_file_path = f'{model0}-{version}-lr{lr}-wd{weight_decay}-bs{batchsize}-p{patience}-{exp_method}_{log_file_path}'
        summary = f'{log_file_path}\n' \
                  f'{model0}:\t' \
                  f'Version: {version}\t' \
                  f'Dataset: {datasetname}\t' \
                  f'LR: {lr}\t' \
                  f'Weight Decay: {weight_decay}\n' \
                  f'Batchsize: {batchsize}\t' \
                  f'Patience: {patience}\t' \
                  f'TDdroprate: {TDdroprate}\t' \
                  f'BUdroprate: {BUdroprate}\t' \
                  f'Explanation Method: {exp_method}\n'
        start_datetime = datetime.datetime.now()
        print(start_datetime)
        print(summary)
        with open(log_file_path, 'a') as f:
            f.write(f'{start_datetime}\n')
            f.write(f'{summary}\n')
        for iter_num in range(iterations):
            torch.manual_seed(iter_num)
            np.random.seed(iter_num)
            random.seed(iter_num)
            if datasetname in ['NewTwitter', 'NewWeibo']:
                dataset_tuple = load5foldData(datasetname)
                treeDic = None  # loadTree(datasetname)
                for fold_num in range(5):
                    seed = int(f'{iter_num}{fold_num}')
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    random.seed(seed)
                    output = test_GCN(treeDic, dataset_tuple[fold_num * 2], dataset_tuple[fold_num * 2 + 1],
                                      TDdroprate, BUdroprate, lr,
                                      weight_decay, patience, n_epochs, batchsize, datasetname, iter_num,
                                      fold=fold_num, device=device, log_file_path=log_file_path, model_type=model,
                                      split_type=split_type, hidden_size=hidden_size, output_size=output_size,
                                      ebgcn_args=args, exp_method=exp_method,
                                      tokeniser=bert_tokeniser, text_encoder=bert_model)
            elif datasetname in ['NewPHEME']:
                treeDic = None
                for fold_num, (fold_train, fold_test) in enumerate(load9foldData(datasetname, upsample=False)):
                    fold_train, fold_train_labels = fold_train
                    fold_test, fold_test_labels = fold_test
                    fold_train_labels = np.asarray(fold_train_labels)
                    # fold_test_labels = np.asarray(fold_test_labels)
                    classes = np.asarray([0, 1, 2, 3])
                    class_weight = compute_class_weight('balanced', classes=classes, y=fold_train_labels)
                    class_weight = torch.FloatTensor(class_weight)
                    seed = int(f'{iter_num}{fold_num}')
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    random.seed(seed)
                    output = test_GCN(treeDic, fold_test, fold_train, TDdroprate, BUdroprate, lr, weight_decay,
                                      patience, n_epochs, batchsize, datasetname, iter_num, fold=fold_num,
                                      device=device, log_file_path=log_file_path, class_weight=class_weight,
                                      model_type=model, split_type=split_type, hidden_size=hidden_size,
                                      output_size=output_size, ebgcn_args=args, exp_method=exp_method,
                                      tokeniser=bert_tokeniser, text_encoder=bert_model)
        print('End of programme')
        end_datetime = datetime.datetime.now()
        print(end_datetime)
        with open(log_file_path, 'a') as f:
            f.write('End of programme')
            f.write(f'{end_datetime}\n')