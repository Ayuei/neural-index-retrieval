import json

import pytorch_lightning as pl
import torch
from torch import nn
from sentence_transformers import SentenceTransformer
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import lru_cache


class MashQA(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        d = {}

        for k, v in self.encodings[idx].items():
            if k == "label":
                d[k] = torch.tensor([v])
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

    for group in read_dict['data'][0:10]:
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


def prepare_dataset(train_path, val_path):
    #return contexts, questions, answers, sents, sent_starts
    train_contexts, train_questions, train_answers, train_sents, train_sent_starts = read_path(train_path)
    val_contexts, val_questions, val_answers, val_sents, val_sent_starts = read_path(val_path)

    model = SentenceTransformer("all-distilroberta-v1")

    train_sent_encodings = []
    val_sent_encodings = []

    @lru_cache(10)
    def encode(sent):
        return model.encode(sent)

    for train_sent in tqdm(train_sents, desc="Train sent encodes"):
        train_sent_encodings.append(encode(train_sent))

    for val_sent in tqdm(val_sents, desc="Val sent encodes"):
        val_sent_encodings.append(encode(val_sent))

    train_question_encodings = model.encode(train_questions, show_progress_bar=True)
    val_question_encodings = model.encode(val_questions, show_progress_bar=True)

    train_encodings = []
    val_encodings = []

    train_encodings = generate_encodings(train_question_encodings, train_sent_encodings, train_answers, train_sent_starts)
    val_encodings = generate_encodings(val_question_encodings, val_sent_encodings, val_answers, val_sent_starts)

    #train_encodings = {"questions": train_question_encodings, "sents": train_sent_encodings}
    #val_encodings = {"questions": val_question_encodings, "sents": val_sent_encodings}

    train_dataset = MashQA(train_encodings)
    val_dataset = MashQA(val_encodings)

    return {"train": train_dataset, "val": val_dataset}


class RegressionNet(pl.LightningModule):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size*2, 1)
        #self.act = entmax.Entmax15(dim=-2)
        self.loss = nn.BCEWithLogitsLoss()

        self.train_metric = pl.metrics.MeanSquaredError()
        self.val_metric = pl.metrics.MeanSquaredError()

    def forward(self, qas, sents, label=None):
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


if __name__ == "__main__":
    dataset = prepare_dataset("./datasets/mashqa_data/train_webmd_squad_v2_consec.json",
                              "./datasets/mashqa_data/val_webmd_squad_v2_consec.json")

    train_set, val_set = dataset["train"], dataset["val"]

    train_loader = DataLoader(train_set, batch_size=2)
    val_loader = DataLoader(val_set, batch_size=2)

    model = RegressionNet(hidden_size=768)

    trainer = pl.Trainer(gpus=1, precision=16, max_epochs=3)
    trainer.fit(model, train_loader, val_loader)
