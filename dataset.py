from typing import List, Dict
import torch
from torch.utils.data import Dataset

from utils import Vocab


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        # print(self._idx2label)
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fns:
        batch_ids = self.vocab.encode_batch(
            [list(item.get("text").split(" ")) for item in samples],
            self.max_len
        )
        batch_ids = torch.tensor(batch_ids)
        try:
            batch_labels = [item.get("intent") for item in samples]
            batch_labels = [torch.tensor(self.label2idx(label)) for label in batch_labels]
            batch_labels = torch.stack(batch_labels)
        except KeyError:
            pass

        batch_lengths = [torch.tensor(len(item)) for item in batch_ids]
        batch_lengths = torch.tensor(batch_lengths)

        batch_index = [item.get("id") for item in samples]

        return {
            "ids": batch_ids,
            "labels": batch_labels,
            "lengths": batch_lengths,
            "index": batch_index
        }

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    ignore_idx = -100

    def collate_fn(self, samples):
        # TODO: implement collate_fn
        batch_ids = self.vocab.encode_batch(
            [list(item.get("tokens")) for item in samples],
            self.max_len
        )
        batch_ids = torch.tensor(batch_ids)

        batch_labels = [item.get("tags") for item in samples]
        try:
            batch_labels = [
                self.label2idx(label)
                for label_item in batch_labels
                for label in label_item
            ]

            if len(batch_labels) < self.max_len:
                batch_labels += [7] * (self.max_len - len(batch_labels))
            batch_labels = torch.tensor(batch_labels)
        except TypeError:
            pass

        batch_index = [item.get("id") for item in samples]

        return {
            "ids": batch_ids,
            "labels": batch_labels,
            "index": batch_index
        }