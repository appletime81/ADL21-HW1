import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import trange
from model import SeqClassifier
from dataset import SeqClsDataset
from utils import Vocab

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
    # test_data_loader = DataLoader(
    #     datasets["test"],
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     collate_fn=datasets["test"].collate_fn,
    # )

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    epoch_losses = []
    epoch_accs = []
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(
        embeddings,
        int(args.hidden_size),
        int(args.num_layers),
        float(args.dropout),
        int(args.bidirectional),
        150,
    )
    model.to(args.device)

    # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        model.train()
        for batch in train_data_loader:
            ids = batch["ids"].to(args.device)
            labels = batch["labels"].to(args.device)
            pred = model(ids).to(args.device)
            print((pred.argmax(dim=1) == labels))
            loss = criterion(pred, labels).to(args.device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            epoch_accs.append((pred.argmax(dim=1) == labels).float().mean().item())

        # TODO: Evaluation loop - calculate accuracy and save model weights
        model.eval()
        with torch.no_grad():
            for batch in dev_data_loader:
                ids = batch["ids"].to(args.device)
                labels = batch["labels"].to(args.device)
                pred = model(ids).to(args.device)
                loss = criterion(pred, labels).to(args.device)
                epoch_losses.append(loss.item())
                epoch_accs.append((pred.argmax(dim=1) == labels).float().mean().item())

    print(epoch_losses)
    print(epoch_accs)


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
