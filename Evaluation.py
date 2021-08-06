import argparse
import torch
import os
from model.casRel import CasRel
from model.data import load_data, get_pred_iterator
from model.config import Config
from model.evaluate import metric


seed = 226
torch.manual_seed(seed)

parser = argparse.ArgumentParser(description='Model Controller')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--max_epoch', type=int, default=50)
parser.add_argument('--max_len', type=int, default=300)
parser.add_argument('--dataset', default='duie', type=str, help='define your own dataset names')
parser.add_argument("--bert_name", default='bert-chinese-wwm', type=str, help='choose pretrained bert name')
parser.add_argument('--bert_dim', default=768, type=int)
parser.add_argument("--device", default='1')
args = parser.parse_args()
con = Config(args)

if args.device != 'cpu':
    assert args.device.isdigit()
    device = torch.device('cuda:{}'.format(args.device))
else:
    device = torch.device('cpu')

if __name__ == '__main__':
    model = CasRel(con).to(device)
    data_bundle, rel_vocab = load_data(con.train_path, con.test_path, con.rel_path)
    test_data = get_pred_iterator(con, data_bundle.get_dataset('test'), rel_vocab, is_test=True)

    # load checkpoint weight
    # path = "./saved_weights/epoch_28#50_duie_model.pkl"
    # if os.path.exists(path):
    #     print("-" * 5 + "Begin Loading Model" + "-" * 5)
    #     checkpoint = torch.load(path)
    #     model.load_state_dict(checkpoint['model'])
    #     print("-" * 5 + "Finish Loading!" + "-" * 5)

    print("-" * 5 + "Begin Loading Model" + "-" * 5)
    model.load_state_dict(torch.load("./saved_weights/epoch_30#30_duie_model.pkl",))
                                     # map_location={'cuda:1': 'cuda:0'}))
                                     # map_location='cpu'))
    print("-" * 5 + "Finish Loading!" + "-" * 5)

    # test
    model.eval()
    print("-" * 5 + "Testing (triples)" + "-" * 5)
    metric(test_data, rel_vocab, con, model, output=False, select="triples")
    print("-" * 5 + "Testing (relation)" + "-" * 5)
    metric(test_data, rel_vocab, con, model, output=False, select="relation")
    print("-" * 5 + "Testing (entity)" + "-" * 5)
    metric(test_data, rel_vocab, con, model, output=False, select="entity")
    print("-" * 5 + "Testing (subject)" + "-" * 5)
    metric(test_data, rel_vocab, con, model, output=False, select="subject")
    print("-" * 5 + "Testing (object)" + "-" * 5)
    metric(test_data, rel_vocab, con, model, output=False, select="object")
