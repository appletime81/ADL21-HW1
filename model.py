from typing import Dict

import torch
from torch.nn import Embedding


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        self.embed_dim = embeddings.shape[1]
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_class = num_class
        self.rnn = torch.nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=True if self.bidirectional else False,
            batch_first=True
        )
        self.fc = torch.nn.Linear(
            in_features=self.hidden_size * 2 if self.bidirectional else self.hidden_size,
            out_features=self.num_class,
        )
        self.dropout = torch.nn.Dropout(self.dropout)

    @property
    def encoder_output_size(self) -> int:
        return self.hidden_size * 2 if self.bidirectional else self.hidden_size

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        embed = self.embed(batch)
        output, (hidden, cell) = self.rnn(embed)
        if self.rnn.bidirectional:
            # print("bidirectional")
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            # print("not bidirectional")
            hidden = hidden[-1, :, :]
            hidden = self.dropout(hidden)
        pred = self.fc(hidden)
        return pred


class SeqTagger(SeqClassifier):
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        embed = self.embed(batch)
        output, (hidden, cell) = self.rnn(embed)
        if self.rnn.bidirectional:
            # print("bidirectional")
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            # print("not bidirectional")
            hidden = hidden[-1, :, :]
            hidden = self.dropout(hidden)
        pred = self.fc(hidden)
        return pred
