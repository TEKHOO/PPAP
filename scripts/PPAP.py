import math
import torch
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl
from einops import rearrange
from torch import einsum
from einops.layers.torch import Rearrange
from torch_geometric.utils import softmax
from torchmetrics import R2Score
from typing import Optional
import torch.nn.functional as F


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    return scatter_sum(src, index, dim, out, dim_size)


class PPAPLayer(nn.Module):
    def __init__(self, d_node = 1280, d_edge = 505, n_heads = 20, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.att_emb = 1280
        self.d_head = self.att_emb // n_heads
        self.dropout = nn.Dropout(dropout)

        self.att_mlp_c = nn.Sequential(
            nn.Linear(d_edge + d_node*2, n_heads),
            nn.ReLU(),
        )

        # (h_j) to d_emb
        self.node_mlp_c = nn.Sequential(
            nn.Linear(d_node * 2 + d_edge, self.att_emb),
            nn.GELU(),
            Rearrange("n (h d) -> n h d", h=n_heads, d=self.d_head),
        )

        self.ff_g = nn.Sequential(
            nn.Linear(self.att_emb, self.att_emb * 4), nn.ReLU(), nn.Linear(self.att_emb * 4, self.att_emb)
        )
        self.to_h_c = nn.Linear(self.att_emb, 1280, bias=False)

    def forward(self, h, edge_core, core_edge_idx):
        # h: (# nodes, d_emb)
        # e: (# edges, d_emb)
        # edge_index: (2, # edges)

        #
        # Global_feature update
        #
        hi_c, hj_c = h[core_edge_idx[0]], h[core_edge_idx[1]]

        hi_eij_hj_c = torch.cat([hi_c, edge_core, hj_c], dim=-1)

        # Compute attention weights for each edge.
        w_c = self.att_mlp_c(hi_eij_hj_c) / math.sqrt(self.d_head)
        att_c = F.softmax(w_c, dim=0)  # n_c h

        # Compute node values.
        vj_c = self.node_mlp_c(hi_eij_hj_c)  # n_c, n_heads, d_head

        # Aggregate node values with attention weights
        # to update node features.
        _global_fea = einsum("nh,nhd->nhd", att_c, vj_c)
        _global_fea = rearrange(_global_fea, "n h d -> n (h d)")
        _global_fea = _global_fea.sum(dim=0)
        _global_fea = self.ff_g(_global_fea)
        global_fea = self.to_h_c(_global_fea)

        return h, edge_core, global_fea, att_c


class PPAP(pl.LightningModule):
    def __init__(
        self,
        d_coreedge=505,
        d_node=5120,
        n_heads=20,
        num_layers=3,
        lr=5e-4,
    ):
        super().__init__()

        # Training parameters
        self.lr = lr
        self.num_layers = num_layers

        self.core_edge_proj = nn.Sequential(
            nn.Linear(d_coreedge, d_coreedge),
            nn.BatchNorm1d(d_coreedge),
        )

        self.layers = nn.ModuleList(
            [
                PPAPLayer(n_heads=n_heads, d_node=d_node, d_edge=d_coreedge)
                for _ in range(self.num_layers)
            ]
        )

        self.to_affinity = nn.Sequential(
            nn.Linear(1280 * num_layers, 100),
            nn.GELU(),
            nn.Linear(100, 1),
        )

        self._init_params()
        self.criterion = nn.MSELoss(reduction="mean")

        # Empty list for validation recovery metrics.
        self.validation_step_outputs = []

        self.r2_score = R2Score()  # 初始化 R2Score 实例

    def _init_params(self):
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, node, core_edge, core_edge_idx):
        core_edge_idx = core_edge_idx.squeeze(0)

        h = node.squeeze(0)
        core_edge = core_edge.squeeze(0)

        e = self.core_edge_proj(core_edge)
        g_fea_record = []
        att_c = 0

        for layer in self.layers:
            h, e, g_fea, att_c = layer(h, e, core_edge_idx)
            g_fea_record.append(g_fea)

        g_fea_record = torch.cat(g_fea_record, dim=-1)

        # 后续尝试核心边flatten输出为一个
        logits = self.to_affinity(g_fea_record)
        return logits, att_c

    def training_step(self, batch):
        node_feat = batch['node_fea']
        core_edge_fea = batch['core_edge_fea']
        core_edge_idx = batch['core_edge_index']
        target = batch['affinity']
        target = target.squeeze(0)

        # node, inner_edge, core_edge, inner_edge_idx, core_edge_idx, batch_idx
        out, _ = self.forward(node_feat, core_edge_fea, core_edge_idx)

        loss = self.criterion(out, target)

        self.log_dict(
            {
                "train/loss": loss,
                # "train/perplexity": torch.exp(loss),
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            #batch_size=self.bsz,
        )

        return loss

    def validation_step(self, batch):
        node_feat = batch['node_fea']
        core_edge_fea = batch['core_edge_fea']
        core_edge_idx = batch['core_edge_index']
        target = batch['affinity']
        target = target.squeeze(0)

        # node, inner_edge, core_edge, inner_edge_idx, core_edge_idx, batch_idx
        out, _ = self.forward(node_feat, core_edge_fea, core_edge_idx)

        loss = self.criterion(out, target)

        self.r2_score.update(torch.tensor(out), torch.tensor(target))  # 更新 R^2值状态

        self.log_dict(
            {
                "val/loss": loss,
                # "val/perplexity": torch.exp(loss),
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            #batch_size=self.bsz,
        )

        return loss

    def on_validation_epoch_end(self):

        r2 = self.r2_score.compute()  # 计算验证集上的 R^2值

        self.validation_step_outputs.append(r2)

        # 使用log或print将R2值输出
        self.log('val_r2', r2, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        print(f'R^2 value: {r2}')
        self.r2_score.reset()  # 最后要记得重置 R^2值状态

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches,
        )

        # return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

