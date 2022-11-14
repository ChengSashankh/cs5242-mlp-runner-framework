import os
import time

from sklearn.metrics import accuracy_score

from analytics.analytics import Analytics
from modules import BasicDeepLearner
import torch


# Model 1
class SimpleMLPTextClassifier(BasicDeepLearner):
    def __init__(self, checkpoint_freq, loss_fn, model_spec, name, sent_cleaner_conf, device, input_dim, batch_size, logger, alpha=1e-4):
        super().__init__(model_spec, alpha, input_dim)
        self.logger = logger
        self.loss_fn = loss_fn
        self.alpha = alpha
        self.checkpoint_freq = checkpoint_freq
        self.name = name
        self.sent_cleaner_conf = sent_cleaner_conf
        self.device = device
        self.train_csv = open(f"outputs/{self.name}/train.csv", "w")
        self.metrics_csv = open(f"outputs/{self.name}/metrics.csv", "w")
        self.start_time = None
        self.end_time = None
        self.bs = batch_size

    def __init_csvs(self):
        self.train_csv.write("epoch,train_loss,val_loss,val_acc\n")
        self.metrics_csv.write("epoch,duration\n")

    def forward(self, X_batch):
        return self.seq(X_batch)

    def save_checkpoint(self, optimizer, curr_epoch, model_path, loss):
        self.logger.log(f"Saving checkpoint to {model_path}")
        torch.save({
            'epoch': curr_epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, model_path)

    def run_training(self, train_x, train_y, val_x, val_y, optim, epochs=10):
        # self.train()
        start_time = time.process_time()

        for e in range(0, epochs):
            _start = time.process_time()
            losses = self._training(train_x, train_y, optim)

            # Save model checkpoint
            mean_loss = torch.tensor(losses).mean()
            if e > 0 and e % self.checkpoint_freq == 0:
                os.makedirs(f"outputs/{self.name}/checkpoints", exist_ok=True)
                self.save_checkpoint(optim, e, f'outputs/{self.name}/checkpoints/epoch-{e}.pt', mean_loss)

            self.cal_loss_and_accuracy(val_x, val_y, mean_loss, e)

            _end = time.process_time()
            self.metrics_csv.write(f"{e},{_end-_start}\n")
            self.metrics_csv.flush()

        Analytics.create_animated_gifs(self.name)
        end_time = time.process_time()
        duration = end_time - start_time
        self.logger.log(f"Training time: {duration}")

    def _training(self, train_x, train_y, optim):
        losses = []
        # for X, Y in zip(torch.Tensor(train_x), torch.Tensor(train_y)):
        for row in range(0, train_x.shape[0], self.bs):
            X = train_x[row: row+self.bs]
            Y = train_y[row: row+self.bs]
            b = X.shape[0]
            Y_preds = self.forward(X)

            loss = self.loss_fn(Y_preds.view(b, 10), Y.view(b))
            losses.append(loss.item())

            optim.zero_grad()
            loss.backward()
            optim.step()

        return losses

    def cal_loss_and_accuracy(self, val_x, val_y, train_loss, epoch_num):
        with torch.no_grad():
            Y_shuffled, Y_preds, losses = [], [], []
            # for X, Y in zip(torch.Tensor(val_x), torch.Tensor(val_y)):
            for i in range(val_x.shape[0]):
                X = val_x[i]
                Y = val_y[i]
                preds = self.forward(X)
                loss = self.loss_fn(preds.view(-1, 10), Y.view(1))
                losses.append(loss.item())

                Y_shuffled.append(Y.view(1))
                Y_preds.append(preds.argmax(dim=-1))

            Y_shuffled = torch.stack(Y_shuffled)
            Y_preds = torch.stack(Y_preds)

            # Check confusion matrix
            Analytics.confusion_matrix_analysis(Y_shuffled, Y_preds, self.name, epoch_num)

            # Check error by class
            Analytics.acc_by_class(Y_shuffled, Y_preds, self.name, epoch_num)

            self.train_csv.write(f"{epoch_num},{train_loss},{torch.tensor(losses).mean()},{accuracy_score(Y_shuffled.detach().numpy(), Y_preds.detach().numpy())}\n")
            self.train_csv.flush()
            self.logger.log("Epoch: {:d} | Train Loss: {:.3f} | Valid Loss : {:.3f} | Valid Acc  : {:.3f}"
                            .format(epoch_num, train_loss, torch.tensor(losses).mean(),
                                    accuracy_score(Y_shuffled.detach().numpy(), Y_preds.detach().numpy())))
