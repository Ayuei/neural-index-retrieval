import torch
from torch.utils.data import RandomSampler


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


# Each will give a list of sentences
# We want each batch to be only 1 context
# Truncate sentences we cannot fit into batch?
# Or delay the gradient accumulation
class SentenceLoader(torch.utils.data.Dataset):
    def __init__(self, encodings, truncate_at=150):
        self.encodings = encodings
        self.truncate_at = truncate_at

    def __getitem__(self, idx):
        d = {}

        for k, v in self.encodings[idx].items():
            if k == "sentence_encode":
                d[k] = torch.stack([torch.tensor(item) for item in v[0:self.truncate_at]])

            if k == "label":
                d[k] = torch.tensor([v])
            elif isinstance(v, str):
                d[k] = v
            else:
                d[k] = torch.tensor(v)

        return d

    def __len__(self):
        return len(self.encodings)


class VariableLengthLoader(object):
    def __init__(self, dataset, target_bsz, drop_last=False):
        self.ds = dataset
        self.my_bsz = target_bsz
        self.drop_last = drop_last
        self.sampler = RandomSampler(dataset)

    def __iter__(self):
        batch = torch.Tensor()
        for idx in self.sampler:
            batch = torch.cat([batch, self.ds[idx]])
            while batch.size()[0] >= self.my_bsz:
                if batch.size()[0] == self.my_bsz:
                    yield batch, False
                    batch = torch.Tensor()
                else:
                    return_batch, batch = batch.split([batch.size()[0]-self.my_bsz])
                    yield return_batch, True
        if batch.size()[0] > 0 and not self.drop_last:
            yield batch, False
