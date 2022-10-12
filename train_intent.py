import json
import pickle
import numpy as np
import torch

from utils import Vocab
from typing import Dict
from tqdm import trange
from pathlib import Path
from model import SeqClassifier
from dataset import SeqClsDataset
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {
        split: args.data_dir / f"{split}.json" 
        for split in SPLITS
    }
    data = {
        split: json.loads(path.read_text()) 
        for split, path in data_paths.items()
    }
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
    train_data_loader = DataLoader(
        datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=datasets["train"].collate_fn,
    )
    dev_data_loader = DataLoader(
        datasets["eval"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=datasets["eval"].collate_fn,
    )

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        len(intent2idx),
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
            torch.save(model.state_dict(), args.ckpt_dir / 'intent_model.pt')


    # TODO: Inference on test set


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
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
    # python train_intent.py --device cuda --ckpt_dir ./ckpt/intent/
