import json
import pickle


from typing import Dict
from pprint import pprint
from utils import Vocab
from pathlib import Path
from dataset import SeqClsDataset
from torch.utils.data import DataLoader


TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]
with open("cache/intent/vocab.pkl", "rb") as f:
    vocab: Vocab = pickle.load(f)
max_len = 128
cache_dir = Path("cache/intent")


# print vocab item
# print(vocab.__dict__)
# open json file
with open("cache/intent/intent2idx.json") as f:
    intent2idx = json.load(f)


data_paths = {split: Path(f"data/intent/{split}.json") for split in SPLITS}
data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}


datasets: Dict[str, SeqClsDataset] = {
    split: SeqClsDataset(split_data, vocab, intent2idx, max_len)
    for split, split_data in data.items()
}


train_loader = DataLoader(
    datasets["train"],
    batch_size=2,
    shuffle=True,
    collate_fn=datasets["train"].collate_fn,
)
for batch in train_loader:
    print(batch.get("encode_sentences"), len(batch.get("encode_sentences")[0][0]))
    print("--------------------------")
    print(batch)
    break
