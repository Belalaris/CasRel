import torch
import os
import json
from transformers import BertTokenizer
from tqdm import tqdm

device = torch.device('cuda:1')


def to_tuple(triple_list, select=""):
    if select == "":
        print("select evaluate mode: triples|relation|entity|subject|object")
        exit(12)
    elif select == "triples":
        ret = set()
        for triple in triple_list:
            ret.add((triple['subject'], triple['predicate'], triple['object']))
    elif select == "relation":
        ret = set()
        for triple in triple_list:
            ret.add(triple['predicate'])
    elif select == 'entity':
        ret = set()
        for triple in triple_list:
            ret.add((triple['subject'], triple['object']))
    elif select == 'subject':
        ret = set()
        for triple in triple_list:
            ret.add(triple['subject'])
    elif select == 'object':
        ret = set()
        for triple in triple_list:
            ret.add(triple['object'])
    return ret


def metric(data_iter, rel_vocab, config, model, output=True, select="", h_bar=0.5, t_bar=0.5):

    triple_order = ['subject', 'relation', 'object']
    entity_order = ['subject', 'object']
    correct_num, predict_num, gold_num = 0, 0, 0
    tokenizer = BertTokenizer.from_pretrained(config.bert_name)

    for batch_x, batch_y in tqdm(data_iter):
        with torch.no_grad():
            # process batch_y
            gold_set = to_tuple(batch_y['triples'][0], select=select)

            # process batch_x
            token_ids = batch_x['token_ids']
            mask = batch_x['mask']
            encoded_text = model.get_encoded_text(token_ids, mask)
            pred_sub_heads, pred_sub_tails = model.get_subs(encoded_text)
            sub_heads = torch.where(pred_sub_heads[0] > h_bar)[0]
            sub_tails = torch.where(pred_sub_tails[0] > t_bar)[0]
            subjects = []
            for sub_head in sub_heads:
                sub_tail = sub_tails[sub_tails >= sub_head]
                if len(sub_tail) > 0:
                    sub_tail = sub_tail[0]
                    subject = ''.join(tokenizer.decode(token_ids[0][sub_head: sub_tail + 1]).split())
                    subjects.append((subject, sub_head, sub_tail))

            if subjects:
                triple_list = []
                repeated_encoded_text = encoded_text.repeat(len(subjects), 1, 1)
                sub_head_mapping = torch.zeros((len(subjects), 1, encoded_text.size(1)), dtype=torch.float,
                                               device=device)
                sub_tail_mapping = torch.zeros((len(subjects), 1, encoded_text.size(1)), dtype=torch.float,
                                               device=device)
                for subject_idx, subject in enumerate(subjects):
                    sub_head_mapping[subject_idx][0][subject[1]] = 1
                    sub_tail_mapping[subject_idx][0][subject[2]] = 1
                pred_obj_heads, pred_obj_tails = model.get_objs_for_specific_sub(sub_head_mapping, sub_tail_mapping,
                                                                                 repeated_encoded_text)
                for subject_idx, subject in enumerate(subjects):
                    sub = subject[0]
                    obj_heads = torch.where(pred_obj_heads[subject_idx] > h_bar)
                    obj_tails = torch.where(pred_obj_tails[subject_idx] > t_bar)
                    for obj_head, rel_head in zip(*obj_heads):
                        for obj_tail, rel_tail in zip(*obj_tails):
                            if obj_head <= obj_tail and rel_head == rel_tail:
                                rel = rel_vocab.to_word(int(rel_head))
                                obj = ''.join(tokenizer.decode(token_ids[0][obj_head: obj_tail + 1]).split())
                                triple_list.append((sub, rel, obj))
                                break
                pred_set = set()
                for s, r, o in triple_list:
                    if select == "triples":
                        pred_set.add((s, r, o))
                    elif select == "relation":
                        pred_set.add(r)
                    elif select == 'entity':
                        pred_set.add((s, o))
                    elif select == 'subject':
                        pred_set.add(s)
                    elif select == 'object':
                        pred_set.add(o)
            else:
                pred_set = set()

            correct_num += len(pred_set & gold_set)
            predict_num += len(pred_set)
            gold_num += len(gold_set)

            if output:
                if not os.path.exists(config.result_dir):
                    # os.mkdir(config.result_dir)
                    os.makedirs(config.result_dir)  # Create a recursive directory tree
                path = os.path.join(config.result_dir,
                                    select + "_" + config.result_save_name)
                fw = open(path, 'a')
                if select == "triples":
                    result = json.dumps({
                        'triple_list_gold':
                            [dict(zip(triple_order, triple)) for triple in gold_set],
                        'triple_list_pred':
                            [dict(zip(triple_order, triple)) for triple in pred_set],
                        'new':
                            [dict(zip(triple_order, triple)) for triple in pred_set - gold_set],
                        'lack':
                            [dict(zip(triple_order, triple)) for triple in gold_set - pred_set]
                    }, ensure_ascii=False)
                elif select == "entity":
                    result = json.dumps({
                        'entity_list_gold':
                            [dict(zip(entity_order, binary)) for binary in gold_set],
                        'entity_list_pred':
                            [dict(zip(entity_order, binary)) for binary in pred_set],
                        'new':
                            [dict(zip(entity_order, binary)) for binary in pred_set - gold_set],
                        'lack':
                            [dict(zip(entity_order, binary)) for binary in gold_set - pred_set]
                    }, ensure_ascii=False)
                elif select == "relation" or "subjet" or "object":
                    result = json.dumps({
                        'gold': list(gold_set),
                        'pred': list(pred_set),
                        'new': list(pred_set - gold_set),
                        'lack': list(gold_set - pred_set)
                    }, ensure_ascii=False)
                fw.write(result + '\n')

    print("correct_num: {}, predict_num: {}, gold_num: {}".format(correct_num, predict_num, gold_num))
    precision = correct_num / (predict_num + 1e-10)
    recall = correct_num / (gold_num + 1e-10)
    f1_score = 2 * precision * recall / (precision + recall + 1e-10)
    print('f1: {:5.4f}, precision: {:5.4f}, recall: {:5.4f}'.format(f1_score, precision, recall))
    return precision, recall, f1_score