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
            bidirectional=self.bidirectional,
        )
        self.fc = torch.nn.Linear(
            in_features=self.hidden_size * 2 if self.bidirectional else self.hidden_size,
            out_features=self.num_class,
        )

    @property
    def encoder_output_size(self) -> int:
        return self.hidden_size * 2 if self.bidirectional else self.hidden_size

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        embed = self.embed(batch["encode_sentences"])
        output, _ = self.rnn(embed)
        output = self.fc(output)
        return output




class SeqTagger(SeqClassifier):
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        raise NotImplementedError


if __name__ == "__main__":
    embeddings = torch.load("cache/intent/embeddings.pt")
    seq_classifier = SeqClassifier(
        embeddings=embeddings,
        hidden_size=512,
        num_layers=2,
        dropout=0.1,
        bidirectional=True,
        num_class=150,
    )

    # see the model architecture
    print(seq_classifier)
    print(seq_classifier.encoder_output_size)