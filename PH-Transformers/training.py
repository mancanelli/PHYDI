import torch
import math
import time
import wandb

class Trainer():
    def __init__(self, net, optimizer, scheduler, criterion, epochs, 
                 threshold, device=True, checkpoint_folder="./checkpoints",
                 lr=0.1, momentum=0.9, weight_decay=5e-4):

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.threshold = threshold
        self.device = device
        self.criterion = criterion
        self.checkpoints_folder = checkpoint_folder
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.bptt = 35
        self.net = net
    
    def _get_batch(self, source, i):
        seq_len = min(self.bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].view(-1)
        return data, target

    def train1(self, train_data, val_data, ntokens):
        for epoch in range(self.epochs):
            epoch_start_time = time.time()

            # training
            self.net.train()
            total_loss = 0.

            for i in range(0, train_data.size(0) - 1, self.bptt):
                data, targets = self._get_batch(train_data, i)
                self.optimizer.zero_grad()
                output = self.net(data)
                loss = self.criterion(output.view(-1, ntokens), targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()
                total_loss += loss.item()
            
            train_loss = total_loss / (len(train_data) - 1)

            # evaluating
            self.net.eval()
            total_loss = 0.

            with torch.no_grad():
                for i in range(0, val_data.size(0) - 1, self.bptt):
                    data, targets = self._get_batch(val_data, i)
                    output = self.net(data)
                    output_flat = output.view(-1, ntokens)
                    total_loss += len(data) * self.criterion(output_flat, targets).item()
            
            val_loss = total_loss / (len(val_data) - 1)
            val_ppl = math.exp(val_loss)

            # printing results
            epoch_end_time = time.time()
            self.scheduler.step()

            print((f"[Epoch: {epoch}][Train loss: {train_loss:.3f}][Val loss: {val_loss:.3f}]"
                   f"[Val ppl: {val_ppl:.3f}][Time: {(epoch_end_time - epoch_start_time):.3f}s]"))
            
            wandb.log({"train loss": train_loss})
            wandb.log({"val loss": val_loss})
            wandb.log({"val ppl": val_ppl})

            if val_ppl <= self.threshold:
                break
