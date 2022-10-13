import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)
    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data = json.loads(args.test_file.read_text())

    data_eval = [
        {
            "token": token,
            "id": item["id"]
        }
        for item in data
        for token in item["tokens"]
    ]

    dataset = SeqTaggingClsDataset(data_eval, vocab, tag2idx, args.max_len)

    # TODO: crecate DataLoader for test dataset
    test_data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    test_model = SeqTagger(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        len(tag2idx),
    )

    test_model.load_state_dict(torch.load(args.ckpt_dir / "slot_model.pt"))
    test_model.to(args.device)
    test_model.eval()

    result_index_list = list()
    result_ids_list = list()

    for batch in test_data_loader:
        ids = batch["ids"].to(args.device)
        pred = test_model(ids).to(args.device)
        result_ids_list.extend(
            pred.argmax(dim=1).tolist()
        )
        result_index_list.extend(batch["index"])

    result_dict = dict()
    for index, id in zip(result_index_list, result_ids_list):
        if index not in result_dict:
            result_dict[index] = []
        result_dict[index].append(
            dataset.idx2label(id)
        )

    csv_dict = {
        "id": [],
        "tags": []
    }
    for k, v in result_dict.items():
        csv_dict["id"].append(k)
        csv_dict["tags"].append(
            " ".join(v)
        )

    csv_result = pd.DataFrame(csv_dict)
    csv_result.to_csv(args.pred_file, index=False)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
    # command
    # python test_slot.py --test_file ./data/slot/test.json --pred_file pred.slot.csv --device cuda
