from fastNLP import Callback
import os
from model.evaluate import metric
import torch


class MyCallBack(Callback):
    def __init__(self, data_iter, rel_vocab, config):
        super().__init__()
        self.best_epoch = 0
        self.best_recall = 0
        self.best_precision = 0
        self.best_f1_score = 0

        self.data_iter = data_iter
        self.rel_vocab = rel_vocab
        self.config = config

    def logging(self, s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            if not os.path.exists(self.config.save_logs_dir):
                os.mkdir(self.config.save_logs_dir)
            with open(os.path.join(self.config.save_logs_dir, self.config.log_save_name), 'a') as f_log:
                f_log.write(s + '\n')

    def on_train_begin(self):
        self.logging("-" * 5 + "Begin Training" + "-" * 5)

    def on_epoch_end(self):
        precision, recall, f1_score = metric(self.data_iter, self.rel_vocab, self.config, self.model,
                                             output=False, select="triples")
        self.logging('epoch {:2d}, f1: {:5.4f}, precision: {:5.4f}, recall: {:5.4f}'
                     .format(self.epoch, f1_score, precision, recall))

        if f1_score > self.best_f1_score:
            self.best_f1_score = f1_score
            self.best_epoch = self.epoch
            self.best_precision = precision
            self.best_recall = recall
            self.logging("Saving the model, epoch: {:2d}, best f1: {:5.4f}, precision: {:5.4f}, recall: {:5.4f}".
                         format(self.best_epoch, self.best_f1_score, precision, recall))
            path = os.path.join(self.config.save_weights_dir, self.config.dataset + "/")
            if not os.path.exists(path):
                os.makedirs(path)
            # save the best checkpoint
            checkpoint = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                          "epoch": self.epoch}
            torch.save(checkpoint, path + "epoch_" + str(self.epoch) + "_" + self.config.weights_save_name)

    def on_train_end(self):
        self.logging("-" * 5 + "Finish training" + "-" * 5)
        self.logging("best epoch: {:2d}, best f1: {:5.4f}, precision: {:5.4f}, recall: {:5.4f}".
                     format(self.best_epoch, self.best_f1_score, self.best_precision, self.best_recall))

