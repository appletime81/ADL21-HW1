import json
import time
import torch
import pickle

import numpy as np
from utils import Vocab
from tqdm import trange
from typing import Dict
from pathlib import Path
from model import SeqTagger
from torch.utils.data import DataLoader
from dataset import SeqTaggingClsDataset
from argparse import ArgumentParser, Namespace


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)
    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    train_data_file = args.data_dir / "train.json"
    train_data = json.loads(train_data_file.read_text())
    train_dataset = SeqTaggingClsDataset(train_data, vocab, tag2idx, args.max_len)

    eval_data_file = args.data_dir / "eval.json"
    eval_data = json.loads(eval_data_file.read_text())
    eval_dataset = SeqTaggingClsDataset(eval_data, vocab, tag2idx, args.max_len)

    # TODO: crecate DataLoader for test dataset
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    eval_data_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=eval_dataset.collate_fn,
    )
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
    criterion = torch.nn.NLLLoss()

    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []
    best_valid_acc = float('-inf')

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        start_time = time.time()
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
            for batch in eval_data_loader:
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

        if epoch_valid_acc > best_valid_acc:
            best_valid_acc = epoch_valid_acc
            torch.save(model.state_dict(), args.ckpt_dir / 'slot_model.pt')
        print(f"Epoch time: {(time.time() - start_time) / 60} min")


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
    # python train_slot.py --device cuda --num_epoch 20 --batch_size 1 --max_len 40 --num_layers 4
