import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import trange

from dataset import SeqClsDataset
from utils import Vocab
from pprint import pprint
TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

with open("cache/slot/vocab.pkl", "rb") as f:
    vocab: Vocab = pickle.load(f)

pprint(vocab.__dict__)


