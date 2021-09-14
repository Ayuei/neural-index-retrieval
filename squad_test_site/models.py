import entmax
import pytorch_lightning as pl
import torch
from sentence_transformers.util import batch_to_device
from torch import nn
from torch.nn import MultiheadAttention


class BaseNetwork(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def training_step(self, train_batch, batch_idx):
        q, s, y = train_batch['question_encode'], train_batch['sentence_encode'], train_batch['label']
        loss, y_hat = self.forward(q, s, y)

        self.log('train_loss', loss)
        self.train_metric(y_hat.detach(), y.detach())
        self.log('train_met', self.train_metric, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        q, s, y = val_batch['question_encode'], val_batch['sentence_encode'], val_batch['label']
        loss, y_hat = self.forward(q, s, y)

        self.val_metric(y_hat.detach(), y.detach())
        self.log('val_loss', loss)
        self.log('val_met', self.val_metric, on_epoch=True, prog_bar=True)


class RegressionNet(BaseNetwork):
    def __init__(self, hidden_size, encoder=None):
        super().__init__()
        self.dense = nn.Linear(hidden_size*2, 1)
        self.encoder = encoder
        self.act = entmax.Entmax15(dim=-2)
        self.criterion = nn.BCEWithLogitsLoss()

        self.train_metric = pl.metrics.MeanSquaredError()
        self.val_metric = pl.metrics.MeanSquaredError()
        self.test_metric = pl.metrics.MeanSquaredError()
        self.test_metric2 = pl.metrics.Accuracy()

    def get_encodes(self, sentences):
        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'): #Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        sentences_batch = sentences
        features = self.encoder.tokenize(sentences_batch)
        features = batch_to_device(features, self.device)

        out_features = self.encoder.forward(features)
        embeddings = out_features['sentence_embedding']

        if input_was_string:
            embeddings = embeddings[0]

        return embeddings

    def forward(self, qas, sents, label=None):
        if self.encoder:
            qas = self.get_encodes(qas)
            sents = self.get_encodes(sents)

        x = torch.cat((qas, sents), dim=-1)
        x = torch.flatten(x, 1)
        x = self.act(x)
        y_hat = self.dense(x)

        if label is None:
            return y_hat
        else:
            loss = self.criterion(y_hat, label)
            return loss, y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        return optimizer

    def test_step(self, test_batch, batch_idx):
        q, s, y = test_batch['question_encode'], test_batch['sentence_encode'], test_batch['label']
        loss, y_hat = self.forward(q, s, y)

        self.test_metric(y_hat.detach(), y.detach())
        self.log('test_loss', loss)
        self.log('test_met', self.test_metric, on_epoch=True, prog_bar=True)

        self.test_acc(torch.round(y_hat), y.int())
        self.log('test_met2', self.test_acc, on_epoch=True, prog_bar=True)


class SelfAttentionNetMultiLabel(BaseNetwork):
    def __init__(self, hidden_size, num_sentences_per_batch, include_impossible=False, encoder=None):
        super().__init__()
        # First neuron will be for predicting if it is impossible
        self.include_impossible = include_impossible
        self.dense = nn.Linear(hidden_size*num_sentences_per_batch, num_sentences_per_batch+int(include_impossible))
        self.encoder = encoder
        self.act = entmax.Entmax15(dim=1)
        self.criterion = nn.BCEWithLogitsLoss()
        self.self_attn = MultiheadAttention(768, 12, dropout=0.1, batch_first=True)

        self.train_metric = pl.metrics.F1(multilabel=True)
        self.val_metric = pl.metrics.F1(multilabel=True)
        self.test_metric = pl.metrics.F1(multilabel=True)
        self.test_metric2 = pl.metrics.Precision(multilabel=True)

    def forward(self, qas, sents, label=None):
        x = torch.stack((qas, sents), dim=1)
        x = self.self_attn(x, x, x)
        x = torch.flatten(x, 1)
        x = self.act(x)
        y_hat = self.dense(x)

        if label is None:
            return y_hat
        else:
            loss = self.criterion(y_hat, label)
            return loss, y_hat

    def get_encodes(self, sentences):
        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'): #Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        sentences_batch = sentences
        features = self.encoder.tokenize(sentences_batch)
        features = batch_to_device(features, self.device)

        out_features = self.encoder.forward(features)
        embeddings = out_features['sentence_embedding']

        if input_was_string:
            embeddings = embeddings[0]

        return embeddings

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        return optimizer

    def test_step(self, test_batch, batch_idx):
        q, s, y = test_batch['question_encode'], test_batch['sentence_encode'], test_batch['label']
        loss, y_hat = self.forward(q, s, y)

        self.test_metric(y_hat.detach(), y.detach())
        self.log('test_loss', loss)
        self.log('test_met', self.test_metric, on_epoch=True, prog_bar=True)

        self.test_acc(y_hat.detach(), y.detach())
        self.log('test_met2', self.test_acc, on_epoch=True, prog_bar=True)
