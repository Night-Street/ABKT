import numpy as np
import torch
from torch import nn


class ItemEmbeddingLayer(nn.Module):
    def __init__(self,
                 k_hidden_size,
                 skill_num,
                 user_num,
                 device):
        super(ItemEmbeddingLayer, self).__init__()
        self.skill_embedding = nn.Parameter(torch.randn(user_num, skill_num, k_hidden_size) * .01)

        self.k_hidden_size = k_hidden_size
        self.skill_num = skill_num
        self.device = device

    def forward(self, user: int, Q_matrix: torch.Tensor, items: torch.Tensor):
        length = len(items)
        items = items.tolist()
        ret = torch.zeros((length, self.k_hidden_size), device=self.device)
        for pos in range(len(items)):
            item = items[pos]
            for skill_idx in range(len(Q_matrix[item])):
                if Q_matrix[item, skill_idx].data == 1:
                    ret[pos] = ret[pos] + self.skill_embedding[user, skill_idx]
        return ret


class PositionalEncoding(nn.Module):
    def __init__(self):
        super(PositionalEncoding, self).__init__()


class SAINT(nn.Module):
    def __init__(self,
                 k_hidden_size: int,
                 skill_num: int,
                 user_num: int,
                 item_num: int,
                 Q_matrix: torch.Tensor,
                 device: str):
        super(SAINT, self).__init__()
        self.transformer = nn.Transformer(d_model=k_hidden_size,
                                          nhead=1,
                                          batch_first=True)
        self.item_embedding_layer = ItemEmbeddingLayer(k_hidden_size, skill_num, user_num, device)
        self.answer_embedding_layer = nn.Parameter(torch.randn(user_num, 2, k_hidden_size) * .01)
        self.start_token = nn.Parameter(torch.zeros(user_num, k_hidden_size) * .01)
        self.predicting_layer = nn.Sequential(
            nn.Linear(k_hidden_size, 1),
            nn.Sigmoid()
        )

        self.k_hidden_size = k_hidden_size
        self.skill_num = skill_num
        self.user_num = user_num
        self.item_num = item_num
        self.Q_matrix = Q_matrix
        self.device = device

    def forward(self,
                user: int,
                items: torch.Tensor,
                answers: torch.Tensor):
        length = len(items)
        items = items.long()
        answers = answers.long()

        item_embeddings = self.item_embedding_layer(user, self.Q_matrix, items)
        item_embeddings = item_embeddings.unsqueeze(0)
        answer_embedding = self.answer_embedding_layer[user, answers]
        answer_embedding = torch.cat([self.start_token[user].unsqueeze(0), answer_embedding[:-1, :]], dim=0)
        answer_embedding = answer_embedding.unsqueeze(0)
        # print(item_embeddings.shape)
        # print(answer_embedding.shape)
        mask = torch.from_numpy(
            np.triu(
                np.ones((length, length)),
                k=1
            ).astype('bool')
        ).to(self.device)
        out = self.transformer(src=item_embeddings,
                               tgt=answer_embedding,
                               src_mask=mask,
                               tgt_mask=mask,
                               memory_mask=mask)
        out = self.predicting_layer(out.squeeze(0))
        return out.squeeze(-1)


if __name__ == '__main__':
    saint = SAINT(5, 30, 12, 120, torch.zeros((100, 30)), 'cpu')
    out = saint.forward(3, torch.Tensor([5, 5, 5, 5]), torch.Tensor([1, 0, 1, 0]))
    print(out)
