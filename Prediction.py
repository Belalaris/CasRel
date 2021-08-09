import argparse
import os
from model.casRel import CasRel
from model.config import Config
from model.predict import *
import json


seed = 226
torch.manual_seed(seed)

parser = argparse.ArgumentParser(description='Model Controller')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--max_epoch', type=int, default=25)
parser.add_argument('--max_len', type=int, default=300)
parser.add_argument('--dataset', default='duie', type=str, help='define your own dataset names')
parser.add_argument("--bert_name", default='bert-chinese-wwm', type=str, help='choose pretrained bert name')
parser.add_argument('--bert_dim', default=768, type=int)
parser.add_argument("--device", default='0')
args = parser.parse_args()
con = Config(args)

if args.device != 'cpu':
    assert args.device.isdigit()
    device = torch.device('cuda:{}'.format(args.device))
else:
    device = torch.device('cpu')

if __name__ == '__main__':
    model = CasRel(con).to(device)
    rel_vocab = load_rel(con.rel_path)

    # load checkpoint weight
    # path = "./saved_weights/duie/epoch_25_duie_model.pkl"
    # if os.path.exists(path):
    #     print("-" * 5 + "Begin Loading Model" + "-" * 5)
    #     checkpoint = torch.load(path)
    #     model.load_state_dict(checkpoint['model'])
    #     print("-" * 5 + "Finish Loading!" + "-" * 5)

    print("-" * 5 + "Begin Loading Model" + "-" * 5)
    path = "./saved_weights/epoch_30#30_duie_model.pkl"
    model.load_state_dict(torch.load(path, ))
                                     # map_location={'cuda:1': 'cuda:0'}))
                                     # map_location='cpu'))
    print("-" * 5 + "Finish Loading!" + "-" * 5)

    # predict
    model.eval()
    while 1:
        text = input("enter a sentence: ")
        tokens = get_tokenized(con, text)
        pred_dict = predictor(tokens, rel_vocab, model)
        pred_result = json.dumps(pred_dict, ensure_ascii=False, indent=4)
        print("predicated SRO:\n", str(pred_result))
