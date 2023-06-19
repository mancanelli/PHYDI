import torch
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

class DatasetHandler1():
    def __init__(self, max_tokens, batch_size, eval_batch_size, device):
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device = device

        self.tokenizer = None
        self.vocab = None

    def getData(self):
        train_iter = WikiText2(split='train')
        self.tokenizer = get_tokenizer('basic_english')
        
        self.vocab = build_vocab_from_iterator(map(self.tokenizer, train_iter), specials=['<unk>'], max_tokens=self.max_tokens)
        self.vocab.set_default_index(self.vocab['<unk>'])

        train_iter, val_iter, _ = WikiText2()
        train_data = self.data_process(train_iter)
        val_data = self.data_process(val_iter)

        train_data = self.batchify(train_data, self.batch_size)
        val_data = self.batchify(val_data, self.eval_batch_size)

        return train_data, val_data

    def data_process(self, raw_text_iter):
        data = [torch.tensor(self.vocab(self.tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    def batchify(self, data, batch_size):
        seq_len = data.size(0) // batch_size
        data = data[:seq_len * batch_size]
        data = data.view(batch_size, seq_len).t().contiguous()
        return data.to(self.device)
