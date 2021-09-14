import json
import pytorch_lightning as pl

from sentence_transformers import SentenceTransformer
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import lru_cache

from cacher import cache_item
from datasets import MashQA
from models import RegressionNet, SelfAttentionNetMultiLabel

TEST_FLAG = True


def read_path(path):
    path = Path(path)
    with open(path, 'rb') as f:
        read_dict = json.load(f)

    contexts = []
    questions = []
    sents = []
    sent_starts = []
    answers = []
    impossible = []

    read_dict = read_dict['data'][0:10] if TEST_FLAG else read_dict['data']

    for group in read_dict:
        for passage in group['paragraphs']:
            context = passage['context']
            s = passage["sent_list"]
            ss = passage["sent_starts"]

            for qa in passage['qas']:
                question = qa['question']
                is_impossible = passage["is_impossible"]
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer["answer_starts"])
                    sents.append(tuple(s))
                    sent_starts.append(ss)
                    impossible.append(is_impossible)

    return contexts, questions, answers, sents, sent_starts, impossible


def generate_encodings_batch(question_encodings, sent_encodings, answers, sent_lists, impossible):
    encodings = []

    for qas, sent_encodes, answer_list, sent_list in tqdm(zip(question_encodings, sent_encodings,
                                                              answers, sent_lists), desc="Questions"):
        sent_starts = [idxs[0] for idxs in sent_list]
        starts = [idxs[0] for idxs in answer_list]

        example = {
            "question_encode": qas,
            "sentence_encode": [],
            "labels": []
        }

        for i, sent_start in enumerate(sent_starts):
            if sent_start in starts:
                example["labels"].append(1.0)
            else:
                example["labels"].append(0.0)

            example["sentence_encode"].append(example)

    return encodings


def generate_encodings_singular(question_encodings, sent_encodings, answers, sent_lists):
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


def prepare_dataset(sen_model, path, skip_encode=False, singular=True):
    contexts, questions, answers, sents, sent_starts, impossible = read_path(path)

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

    dset = None

    if singular:
        encodings = generate_encodings_singular(question_encodings, sent_encodings, answers, sent_starts)
        dset = MashQA(encodings)
    else:
        encodings = generate_encodings_batch(question_encodings, sent_encodings, answers, sent_starts, impossible)
        dset = MashQA(encodings)

    return dset


BACKPROP_FLAG = True

if __name__ == "__main__":
    model = SentenceTransformer("all-distilroberta-v1")
    suffix = "backprop" if BACKPROP_FLAG else ""
    suffix += "test" if TEST_FLAG else ""

    train_set = cache_item(prepare_dataset(model, "./datasets/mashqa_data/train_webmd_squad_v2_consec.json",
                                           skip_encode=BACKPROP_FLAG), path="train.set" + suffix)
    val_set = cache_item(prepare_dataset(model, "./datasets/mashqa_data/val_webmd_squad_v2_consec.json",
                                         skip_encode=BACKPROP_FLAG), "val.set" + suffix)
    test_set = cache_item(prepare_dataset(model, "./datasets/mashqa_data/test_webmd_squad_v2_consec.json",
                                          skip_encode=BACKPROP_FLAG), "test.set" + suffix)

    train_loader = DataLoader(train_set, batch_size=2)
    val_loader = DataLoader(val_set, batch_size=2)
    test_loader = DataLoader(val_set, batch_size=2)

    #model = RegressionNet(hidden_size=768, encoder=model if BACKPROP_FLAG else None)
    model = SelfAttentionNetMultiLabel(hidden_size=768, num_sentences_per_batch=150)

    trainer = pl.Trainer(gpus=1, precision=16, max_epochs=1)
    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint("finished.chkpt"+suffix)
    print(trainer.test(dataloaders=test_loader))
