import json

import pytorch_lightning as pl
import torch
from sentence_transformers.util import batch_to_device
from torch import nn
from sentence_transformers import SentenceTransformer
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import lru_cache

TEST_FLAG=False

class MashQA(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        d = {}

        for k, v in self.encodings[idx].items():
            if k == "label":
                d[k] = torch.tensor([v])
            elif isinstance(v, str):
                d[k] = v
            else:
                d[k] = torch.tensor(v)

        return d

    def __len__(self):
        return len(self.encodings)


def read_path(path):
    path = Path(path)
    with open(path, 'rb') as f:
        read_dict = json.load(f)

    contexts = []
    questions = []
    sents = []
    sent_starts = []
    answers = []

    read_dict = read_dict['data'][0:10] if TEST_FLAG else read_dict['data']

    for group in read_dict:
        for passage in group['paragraphs']:
            context = passage['context']
            s = passage["sent_list"]
            ss = passage["sent_starts"]

            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer["answer_starts"])
                    sents.append(tuple(s))
                    sent_starts.append(ss)

    return contexts, questions, answers, sents, sent_starts


def generate_encodings(question_encodings, sent_encodings, answers, sent_lists):
    encodings = []

    for qas, sent_encodes, answer_list, sent_list in tqdm(zip(question_encodings, sent_encodings,
                                                              answers, sent_lists), desc="Questions"):
        sent_starts = [idxs[0] for idxs in sent_list]
        starts = [idxs[0] for idxs in answer_list]

        for i, sent_start in enumerate(sent_starts):
            example = {
                "question_encode": qas,
                "sentence_encode": sent_encodes[i],
                "label": None
            }

            if sent_start in starts:
                example["label"] = 1.0
            else:
                example["label"] = 0.0

            encodings.append(example)

    return encodings


def prepare_dataset(sen_model, path, skip_encode=False):
    contexts, questions, answers, sents, sent_starts = read_path(path)

    question_encodings = questions
    sent_encodings = sents

    if not skip_encode:
        sent_encodings = []

        @lru_cache(10)
        def encode(sen):
            return sen_model.encode(sen)

        for sent in tqdm(sents, desc="Train sent encodes"):
            sent_encodings.append(encode(sent))

        question_encodings = model.encode(questions, show_progress_bar=True)

    encodings = generate_encodings(question_encodings, sent_encodings, answers, sent_starts)

    dset = MashQA(encodings)

    return dset



class RegressionNet(pl.LightningModule):
    def __init__(self, hidden_size, encoder=None):
        super().__init__()
        self.dense = nn.Linear(hidden_size*2, 1)
        self.encoder = encoder
        #self.act = entmax.Entmax15(dim=-2)
        self.loss = nn.BCEWithLogitsLoss()

        self.train_metric = pl.metrics.MeanSquaredError()
        self.val_metric = pl.metrics.MeanSquaredError()
        self.test_metric = pl.metrics.MeanSquaredError()
        self.test_acc = pl.metrics.Accuracy()

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
            #if self.normalize_embeddings:
            #    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        if input_was_string:
            embeddings = embeddings[0]

        return embeddings

    def forward(self, qas, sents, label=None):
        if self.encoder:
            qas = self.get_encodes(qas)
            sents = self.get_encodes(sents)

        x = torch.cat((qas, sents), dim=-1)
        x = torch.flatten(x, 1)
        #x = self.act(x)
        y_hat = self.dense(x)

        if label is None:
            return y_hat
        else:
            loss = self.loss(y_hat, label)
            return loss, y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

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

    def test_step(self, test_batch, batch_idx):
        q, s, y = test_batch['question_encode'], test_batch['sentence_encode'], test_batch['label']
        loss, y_hat = self.forward(q, s, y)

        self.test_metric(y_hat.detach(), y.detach())
        self.log('test_loss', loss)
        self.log('test_met', self.test_metric, on_epoch=True, prog_bar=True)

        self.test_acc(torch.round(y_hat), y.int())
        self.log('test_acc', self.test_acc, on_epoch=True, prog_bar=True)


BACKPROP_FLAG = True

if __name__ == "__main__":
    model = SentenceTransformer("all-distilroberta-v1")

    train_set = prepare_dataset(model, "./datasets/mashqa_data/train_webmd_squad_v2_consec.json",
                                skip_encode=BACKPROP_FLAG)
    val_set = prepare_dataset(model, "./datasets/mashqa_data/val_webmd_squad_v2_consec.json",
                              skip_encode=BACKPROP_FLAG)
    test_set = prepare_dataset(model, "./datasets/mashqa_data/test_webmd_squad_v2_consec.json",
                               skip_encode=BACKPROP_FLAG)

    train_loader = DataLoader(train_set, batch_size=2)
    val_loader = DataLoader(val_set, batch_size=2)
    test_loader = DataLoader(val_set, batch_size=2)

    model = RegressionNet(hidden_size=768, encoder=model if BACKPROP_FLAG else None)

    trainer = pl.Trainer(gpus=1, precision=16, max_epochs=1)
    trainer.fit(model, train_loader, val_loader)
    print(trainer.test(dataloaders=test_loader))
