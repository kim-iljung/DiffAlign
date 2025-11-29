# ===== Standard library =====
import math

# ===== Third-party =====
import torch
from torch import nn


# ---------------- Cross-graph attention (same spirit as original) ----------------


class CrossAttention(nn.Module):
    """Q <- R masked dense cross-attention with optional coordinate updates."""
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.0, coord_update: bool = True):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.dh = dim // heads
        self.coord_update = coord_update

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_r = nn.LayerNorm(dim)
        self.ff_q = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim),
        )

        self.k_null = nn.Parameter(torch.randn(self.heads, self.dh) * 0.02)   # [H,Dh]
        self.v_null = nn.Parameter(torch.randn(self.heads, self.dh) * 0.02)   # [H,Dh]
        self.b_null = nn.Parameter(torch.full((self.heads,), -1.0))           # [H]

        self.coord_edge_mlp = nn.Sequential(
            nn.Linear(2 * dim + 1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
        )
        self.coord_scalar = nn.Linear(dim, 1, bias=False)
        nn.init.xavier_uniform_(self.coord_scalar.weight, gain=1e-2)

    @staticmethod
    def _masked_logsumexp(
        logits: torch.Tensor,
        mask: torch.Tensor,
        dim: int = -1,
        eps: float = 1e-9,
    ) -> torch.Tensor:
        # logits: [..., L], mask: [..., L] (bool)
        neg_inf = torch.finfo(logits.dtype).min
        masked = torch.where(mask, logits, torch.full_like(logits, neg_inf))
        m = torch.amax(masked, dim=dim, keepdim=True)
        sumexp = torch.sum(torch.exp(masked - m), dim=dim, keepdim=True)
        return (m + torch.log(sumexp + eps)).squeeze(dim)

    def preproject_R(self, h_r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        h_r: [B,M,D]  ->  (rh [B,M,D], k_r [B,H,M,Dh], v_r [B,H,M,Dh])
        """
        rh = self.norm_r(h_r)
        k_lin = self.k_proj(rh).contiguous().view(
            rh.size(0), rh.size(1), self.heads, self.dh
        )  # [B,M,H,Dh]
        v_lin = self.v_proj(h_r).contiguous().view(
            h_r.size(0), h_r.size(1), self.heads, self.dh
        )  # [B,M,H,Dh]
        k_r = k_lin.permute(0, 2, 1, 3).contiguous()  # [B,H,M,Dh]
        v_r = v_lin.permute(0, 2, 1, 3).contiguous()  # [B,H,M,Dh]
        return rh, k_r, v_r

    def forward_dense(
        self,
        h_q: torch.Tensor,
        x_q: torch.Tensor,
        h_r: torch.Tensor,
        x_r: torch.Tensor,
        mask_q: torch.Tensor,
        mask_r: torch.Tensor,
        *,
        pre_k: torch.Tensor | None = None,
        pre_v: torch.Tensor | None = None,
        pre_rh: torch.Tensor | None = None,
        coord_chunk_M: int = 256,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        h_q: [B,N,D], x_q: [B,N,3], mask_q: [B,N] (True=valid)
        h_r: [B,M,D], x_r: [B,M,3], mask_r: [B,M]
        """
        B, N, D = h_q.shape
        _, M, _ = h_r.shape
        H, Dh = self.heads, self.dh
        inv_sqrt_dh = 1.0 / math.sqrt(Dh)

        qh = self.norm_q(h_q)                                      # [B,N,D]
        q_lin = self.q_proj(qh).contiguous().view(B, N, H, Dh)     # [B,N,H,Dh]
        q = q_lin.permute(0, 2, 1, 3).contiguous()                 # [B,H,N,Dh]

        if pre_k is None or pre_v is None or pre_rh is None:
            rh, k_r, v_r = self.preproject_R(h_r)
        else:
            rh, k_r, v_r = pre_rh, pre_k, pre_v                    # [B,M,D], [B,H,M,Dh], [B,H,M,Dh]

        mq = mask_q[:, None, :, None]       # [B,1,N,1]
        mr = mask_r[:, None, None, :]       # [B,1,1,M]
        pair_mask = mq & mr                  # [B,1,N,M]

        # logits (real)
        logits_real = torch.einsum(
            "bhnd,bhdm->bhnm",
            q,
            k_r.transpose(-1, -2),
        ) * inv_sqrt_dh  # [B,H,N,M]
        neg_inf = torch.finfo(logits_real.dtype).min
        logits_real = torch.where(
            pair_mask,
            logits_real,
            torch.full_like(logits_real, neg_inf),
        )

        # logits (null)
        logits_null = torch.einsum("bhnd,hd->bhn", q, self.k_null) * inv_sqrt_dh  # [B,H,N]
        logits_null = logits_null + self.b_null.view(1, H, 1)

        # LogSumExp (real + null)
        lse_real = self._masked_logsumexp(logits_real, pair_mask, dim=-1)     # [B,H,N]
        lse_all = torch.logaddexp(lse_real, logits_null)                      # [B,H,N]

        # attention weights
        alpha_real = torch.exp(logits_real - lse_all[:, :, :, None])          # [B,H,N,M]
        alpha_null = torch.exp(logits_null - lse_all)                          # [B,H,N]
        alpha_real = self.dropout(alpha_real)
        alpha_null = self.dropout(alpha_null)

        msg_real = torch.einsum("bhnm,bhmd->bhnd", alpha_real, v_r)           # [B,H,N,Dh]
        msg_null = alpha_null[:, :, :, None] * self.v_null.view(1, H, 1, Dh)  # [B,H,N,Dh]
        h_msg = (msg_real + msg_null).permute(0, 2, 1, 3).contiguous().view(B, N, D)  # [B,N,D]

        h_out = h_q + self.out_proj(h_msg)
        h_out = h_out + self.ff_q(h_out)

        if (not self.coord_update) or M == 0 or N == 0:
            return h_out, x_q

        p_real = alpha_real.sum(dim=-1).clamp(0.0, 1.0)
        gate_q = p_real.mean(dim=1)
        gate_q = gate_q.unsqueeze(-1)                  # [B,N,1]

        alpha_s = alpha_real.mean(dim=1)  # [B,N,M]
        qh_new = self.norm_q(h_out)       # [B,N,D]
        rh_use = self.norm_r(h_r) if pre_rh is None else pre_rh  # [B,M,D]

        dx = torch.zeros_like(x_q)        # [B,N,3]
        chunk = max(1, int(coord_chunk_M))
        for m0 in range(0, M, chunk):
            m1 = min(M, m0 + chunk)
            mr_chunk = mask_r[:, m0:m1]               # [B,m]
            if not torch.any(mr_chunk):
                continue

            diff = x_q[:, :, None, :] - x_r[:, None, m0:m1, :]
            radial = (diff * diff).sum(dim=-1, keepdim=True)          # [B,N,m,1]
            dirn = diff * torch.rsqrt(radial + 1e-8)                  # [B,N,m,3]

            qh_blk = qh_new[:, :, None, :].expand(-1, -1, m1 - m0, -1)
            rh_blk = rh_use[:, None, m0:m1, :].expand(-1, N, -1, -1)
            edge_in = torch.cat([qh_blk, rh_blk, radial], dim=-1)
            s = self.coord_scalar(self.coord_edge_mlp(edge_in)).squeeze(-1)  # [B,N,m]

            valid = (mask_q[:, :, None] & mr_chunk[:, None, :])              # [B,N,m]
            w = torch.where(
                valid,
                alpha_s[:, :, m0:m1],
                torch.zeros_like(alpha_s[:, :, m0:m1]),
            )
            s = torch.where(valid, s, torch.zeros_like(s))

            dx += (dirn * (s * w).unsqueeze(-1)).sum(dim=2)

        dx = dx * gate_q  # [B,N,3]

        x_out = x_q + dx
        return h_out, x_out

    def forward(self, h_q, x_q, h_r, x_r, q2r_edge_index=None, k_train: int = None):
        """
        Legacy-compatible interface (kept for callers in DiffAlign).
        """
        mask_q = torch.any(torch.isfinite(h_q), dim=-1)  # [B,N]
        mask_r = torch.any(torch.isfinite(h_r), dim=-1)  # [B,M]
        return self.forward_dense(h_q, x_q, h_r, x_r, mask_q, mask_r)


class CrossGraphAligner(nn.Module):
    """
    Masked dense cross-attention aligner (drop-in).
    """
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.0,
                 coord_update: bool = True,
                 num_layers: int = 6, recompute_each: int = 1,
                 coord_chunk_M: int = 256):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttention(dim=dim, heads=heads, dropout=dropout, coord_update=coord_update)
            for _ in range(num_layers)
        ])
        self.recompute_each = int(recompute_each)
        self.coord_chunk_M = int(coord_chunk_M)

    @staticmethod
    @torch.no_grad()
    def knn_q2r_edges(*args, **kwargs):
        return torch.empty((2, 0), dtype=torch.long, device=kwargs.get('x_q', torch.tensor((), device='cpu')).device if isinstance(kwargs, dict) and 'x_q' in kwargs else 'cpu')

    @staticmethod
    def _pack_by_pair(h, x, batch, graph_idx):
        """Pack per-pair Q/R tensors into padded batches and masks."""
        device = h.device
        qmask = (graph_idx % 2 == 0)
        rmask = ~qmask
        idx_q_all = torch.nonzero(qmask, as_tuple=False).view(-1)
        idx_r_all = torch.nonzero(rmask, as_tuple=False).view(-1)
        if idx_q_all.numel() == 0 or idx_r_all.numel() == 0:
            return None

        pair_ids = torch.unique(batch)
        B = pair_ids.numel()

        q_lists, r_lists = [], []
        maxN = 0; maxM = 0
        for pid in pair_ids:
            q_idx = idx_q_all[(batch[idx_q_all] == pid)]
            r_idx = idx_r_all[(batch[idx_r_all] == pid)]
            q_lists.append(q_idx); r_lists.append(r_idx)
            maxN = max(maxN, q_idx.numel())
            maxM = max(maxM, r_idx.numel())

        D = h.size(-1)
        h_q = torch.zeros((B, maxN, D), device=device, dtype=h.dtype)
        x_q = torch.zeros((B, maxN, 3), device=device, dtype=x.dtype)
        h_r = torch.zeros((B, maxM, D), device=device, dtype=h.dtype)
        x_r = torch.zeros((B, maxM, 3), device=device, dtype=x.dtype)
        mask_q = torch.zeros((B, maxN), device=device, dtype=torch.bool)
        mask_r = torch.zeros((B, maxM), device=device, dtype=torch.bool)

        for b, (q_idx, r_idx) in enumerate(zip(q_lists, r_lists)):
            n, m = q_idx.numel(), r_idx.numel()
            if n > 0:
                h_q[b, :n] = h[q_idx]
                x_q[b, :n] = x[q_idx]
                mask_q[b, :n] = True
            if m > 0:
                h_r[b, :m] = h[r_idx]
                x_r[b, :m] = x[r_idx]
                mask_r[b, :m] = True

        return {
            "pair_ids": pair_ids,
            "q_lists": q_lists, "r_lists": r_lists,
            "h_q": h_q, "x_q": x_q, "mask_q": mask_q,
            "h_r": h_r, "x_r": x_r, "mask_r": mask_r,
        }

    @staticmethod
    def _unpack_Q(h_q_new, x_q_new, pack, h_global, x_global):
        """Unpack padded Q tensors back into the global tensors."""
        for b, q_idx in enumerate(pack["q_lists"]):
            n = q_idx.numel()
            if n == 0:
                continue
            h_global[q_idx] = h_q_new[b, :n]
            x_global[q_idx] = x_q_new[b, :n]
        return h_global, x_global

    def forward(self, h, x, batch, graph_idx):
        pack = self._pack_by_pair(h, x, batch, graph_idx)
        if pack is None:
            return h, x

        h_q = pack["h_q"]; x_q = pack["x_q"]; mask_q = pack["mask_q"]
        h_r = pack["h_r"]; x_r = pack["x_r"]; mask_r = pack["mask_r"]

        pre_list = []
        for layer in self.layers:
            rh, k_r, v_r = layer.preproject_R(h_r)
            pre_list.append((rh, k_r, v_r))

        for li, layer in enumerate(self.layers):
            rh, k_r, v_r = pre_list[li]
            h_q, x_q = layer.forward_dense(
                h_q, x_q, h_r, x_r, mask_q, mask_r,
                pre_k=k_r, pre_v=v_r, pre_rh=rh,
                coord_chunk_M=self.coord_chunk_M,
            )

        h, x = self._unpack_Q(h_q, x_q, pack, h, x)
        return h, x


__all__ = ["CrossAttention", "CrossGraphAligner"]
