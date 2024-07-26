import torch


class PositionalEncoding(torch.nn.Module):

    init_std: float
    pos_embedding: torch.nn.Parameter

    def __init__(self, dim: list[int], init_std: float = 0.2):
        super().__init__()
        self.init_std = init_std
        assert len(dim) == 3
        self.pos_embedding: torch.nn.Parameter = torch.nn.Parameter(
            torch.randn(1, *dim)
        )
        self.init_parameters()

    def init_parameters(self):
        torch.nn.init.trunc_normal_(self.pos_embedding, std=self.init_std)

    def forward(self, input: torch.Tensor):
        return input + self.pos_embedding
