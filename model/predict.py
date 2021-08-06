import torch
import json
from fastNLP import Vocabulary
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence

device = torch.device('cuda:0')

def load_rel(rel_dict_path):
    id2rel = json.load(open(rel_dict_path))
    rel_vocab = Vocabulary(unknown=None, padding=None)
    rel_vocab.add_word_lst(list(id2rel.values()))
    return rel_vocab

def get_tokenized(config, text):
    tokenizer = BertTokenizer.from_pretrained('./bert-chinese-wwm')
    tokenized = tokenizer([text], max_length=config.max_len, truncation=True)
    tokens = tokenized['input_ids']
    masks = tokenized['attention_mask']

    token_ids = torch.tensor(tokens, dtype=torch.long)
    masks = torch.tensor(masks, dtype=torch.bool)

    batch_token_ids = pad_sequence(token_ids, batch_first=True)
    batch_masks = pad_sequence(masks, batch_first=True)

    return {"token_ids": batch_token_ids.to(device),
            "mask": batch_masks.to(device)
            }

def predictor(token_dict, rel_vocab, model, h_bar=0.5, t_bar=0.5):
    tokenizer = BertTokenizer.from_pretrained('./bert-chinese-wwm')
    with torch.no_grad():
        # process tokenized text
        token_ids = token_dict['token_ids']
        mask = token_dict['mask']
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
            pred_dict = {"entity": set(), "relation": [], "relation_type": []}
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
                            pred_dict["entity"].update({sub, obj})
                            pred_dict["relation"].append([sub, obj])
                            pred_dict["relation_type"].append(rel)
                            break

            pred_dict["entity"] = list(pred_dict["entity"])

        else:
            pred_dict = {}
    return pred_dict