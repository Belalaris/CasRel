import argparse
import torch
import torch.optim as optim
import os
from model.casRel import CasRel
from model.callback import MyCallBack
from model.data import load_data, get_data_iterator, get_pred_iterator
from model.config import Config
import torch.nn.functional as F
from fastNLP import Trainer, LossBase


seed = 226
torch.manual_seed(seed)

parser = argparse.ArgumentParser(description='Model Controller')
parser.add_argument('--lr', type=float, default=5e-6, help='learning rate')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--max_epoch', type=int, default=25)
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

class MyLoss(LossBase):
    def __init__(self):
        super(MyLoss, self).__init__()

    def get_loss(self, predict, target):
        mask = target['mask']

        def loss_fn(pred, gold, mask):
            pred = pred.squeeze(-1)
            loss = F.binary_cross_entropy(pred, gold, reduction='none')
            if loss.shape != mask.shape:
                mask = mask.unsqueeze(-1)
            loss = torch.sum(loss * mask) / torch.sum(mask)
            return loss

        return loss_fn(predict['sub_heads'], target['sub_heads'], mask) + \
               loss_fn(predict['sub_tails'], target['sub_tails'], mask) + \
               loss_fn(predict['obj_heads'], target['obj_heads'], mask) + \
               loss_fn(predict['obj_tails'], target['obj_tails'], mask)

    def __call__(self, pred_dict, target_dict, check=False):
        loss = self.get_loss(pred_dict, target_dict)
        return loss


if __name__ == '__main__':
    model = CasRel(con).to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=con.lr)
    epoch = 1

    # load checkpoint weight
    # path = "./saved_weights/duie/.pkl"
    # if os.path.exists(path):
    #     print("-" * 5 + "Begin Loading Model" + "-" * 5)
    #     checkpoint = torch.load(path)
    #     model.load_state_dict(checkpoint['model'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     epoch = checkpoint['epoch']
    #     print("-" * 5 + "Finish Loading!" + "-" * 5)

    data_bundle, rel_vocab = load_data(con.train_path, con.test_path, con.rel_path)
    train_data = get_data_iterator(con, data_bundle.get_dataset('train'), rel_vocab)
    # dev_data is not used
    test_data = get_pred_iterator(con, data_bundle.get_dataset('test'), rel_vocab)

    # train
    model.train()
    # save checkpoint in callbacks
    trainer = Trainer(train_data=train_data, model=model, optimizer=optimizer, loss=MyLoss(),
                      batch_size=con.batch_size, print_every=con.period, use_tqdm=True,
                      n_epochs=con.max_epoch, epoch=epoch,
                      callbacks=MyCallBack(test_data, rel_vocab, con))
    trainer.train()
