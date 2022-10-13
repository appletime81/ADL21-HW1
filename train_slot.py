import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab
import numpy as np

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    # loading embedding
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    # loading mapping table
    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    TRAIN = "train"
    DEV = "eval"
    SPLITS = [TRAIN, DEV]

    # loading data
    data_paths = {
        split: args.data_dir / f"{split}.json"
        for split in SPLITS
    }
    data = {
        split: json.loads(path.read_text())
        for split, path in data_paths.items()
    }
    data_train = [
        {
            "token": token,
            "tag": tag,
            "id": item["id"]

        }
        for item in data["train"]
        for token, tag in zip(item["tokens"], item["tags"])
    ]
    data_eval = [
        {
            "token": token,
            "tag": tag,
            "id": item["id"]

        }
        for item in data["eval"]
        for token, tag in zip(item["tokens"], item["tags"])
    ]
    data = {
        "train": data_train,
        "eval": data_eval
    }

    # generate dataset
    datasets: Dict[str, SeqTaggingClsDataset] = {
        split: SeqTaggingClsDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }

    # generate dataloader
    train_data_loader = DataLoader(
        datasets["train"],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=datasets["train"].collate_fn
    )
    dev_data_loader = DataLoader(
        datasets["eval"],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=datasets["eval"].collate_fn,
    )

    # start training
    model = SeqTagger(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        len(tag2idx),
    )
    model.to(args.device)

    # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []
    best_valid_loss = float('inf')

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        model.train()
        epoch_train_losses = []
        epoch_train_accs = []
        for batch in train_data_loader:
            ids = batch["ids"].to(args.device)
            labels = batch["labels"].to(args.device)
            pred = model(ids).to(args.device)
            loss = criterion(pred, labels).to(args.device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())
            epoch_train_accs.append((pred.argmax(dim=1) == labels).float().mean().item())

        # TODO: Evaluation loop - calculate accuracy and save model weights
        model.eval()
        epoch_eval_losses = []
        epoch_eval_accs = []
        with torch.no_grad():
            for batch in dev_data_loader:
                ids = batch["ids"].to(args.device)
                labels = batch["labels"].to(args.device)
                pred = model(ids).to(args.device)
                loss = criterion(pred, labels).to(args.device)
                epoch_eval_losses.append(loss.item())
                epoch_eval_accs.append((pred.argmax(dim=1) == labels).float().mean().item())

        train_losses.extend(epoch_train_losses)
        train_accs.extend(epoch_train_accs)
        valid_losses.extend(epoch_eval_losses)
        valid_accs.extend(epoch_eval_accs)

        epoch_train_loss = np.mean(epoch_train_losses)
        epoch_train_acc = np.mean(epoch_train_accs)
        epoch_valid_loss = np.mean(epoch_eval_losses)
        epoch_valid_acc = np.mean(epoch_eval_accs)

        print(epoch_train_acc)
        print(epoch_valid_acc)

        if epoch_valid_loss < best_valid_loss:
            best_valid_loss = epoch_valid_loss
            torch.save(model.state_dict(), args.ckpt_dir / 'slot_model.pt')


def parse_args() -> Namespace:
    parser = ArgumentParser()
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

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
    # command
    # python train_slot.py --device cuda --num_epoch 60
