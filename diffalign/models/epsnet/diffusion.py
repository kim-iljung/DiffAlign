# ===== Standard library =====
import math
from typing import Optional

# ===== Third-party =====
import torch
from torch import autograd
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from uff_torch import UFFTorch, build_uff_inputs, merge_uff_inputs

# ===== Local (project) =====
from ..encoder.egnn import EGNN
from ..encoder.edge import MLPEdgeEncoder
from ..encoder.cross_attention import CrossGraphAligner
from ..common import extend_graph_order_radius


# ---------------- Schedules ----------------

def linear_beta_schedule(num_timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)

def cosine_beta_schedule(num_timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Nichol & Dhariwal (2021): https://arxiv.org/abs/2102.09672
    Returns betas of length T (float32).
    """
    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1. - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999).float()


# ---------------- Positional / time encoders ----------------

class SinusoidalPosEmb(nn.Module):
    """Sine/cosine timestep embedding (float32)."""
    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Embedding dimension ({dim}) must be even.")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        t = t.float()
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        pos = t.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat((pos.sin(), pos.cos()), dim=-1).float()


class DDPMTimeEncoder(nn.Module):
    """SinusoidalPosEmb + MLP for timestep embeddings."""
    def __init__(self, embed_dim: int, activation=nn.SiLU):
        super().__init__()
        sine_embed_dim = embed_dim if (embed_dim % 2 == 0) else (embed_dim - 1)
        self.pos_emb = SinusoidalPosEmb(sine_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(sine_embed_dim, embed_dim),
            activation(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.pos_emb(t)).float()


# ---------------- Geometry helpers ----------------

def get_distance(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.empty((0,), dtype=pos.dtype, device=pos.device)
    return (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)


# ---------------- Batch merge ----------------

def _pick_device(*objs) -> torch.device:
    """
    Pick a common device from a list of tensors/Batches.
    Priority: first CUDA device encountered; else CPU.
    """
    for o in objs:
        if isinstance(o, torch.Tensor):
            if o.is_cuda:
                return o.device
        elif isinstance(o, Batch):
            # Try representative attribute
            for attr in ("pos", "x", "atom_type", "edge_index"):
                if hasattr(o, attr) and getattr(o, attr) is not None:
                    t = getattr(o, attr)
                    if isinstance(t, torch.Tensor) and t.is_cuda:
                        return t.device
    return torch.device("cpu")

def merge_graphs_in_batch(batch1: Batch, batch2: Batch, device: Optional[torch.device] = None) -> Batch:
    """
    Merge as [Q1,R1,Q2,R2,...] and attach graph_idx (even=query, odd=ref) and pair-level batch.
    All Data objects are moved to `device` beforehand to avoid CPU/CUDA mixing.
    """
    if device is None:
        device = _pick_device(batch1, batch2)

    data_list = []
    for d1, d2 in zip(batch1.to_data_list(), batch2.to_data_list()):
        data_list.append(d1.to(device))
        data_list.append(d2.to(device))
    if not data_list:
        # empty Batch on target device
        empty = Batch()
        # attach empty required attrs on correct device if needed later
        return empty

    merge_batch = Batch.from_data_list(data_list)  # now all on same device
    num_nodes_list = [d.num_nodes for d in data_list]

    # graph_idx/batch on correct device
    graph_idx_list = [torch.full((n,), i, dtype=torch.long, device=device) for i, n in enumerate(num_nodes_list)]
    batch_idx_list = [torch.full((n,), i // 2, dtype=torch.long, device=device) for i, n in enumerate(num_nodes_list)]

    merge_batch.graph_idx = torch.cat(graph_idx_list) if graph_idx_list else torch.empty(0, dtype=torch.long, device=device)
    merge_batch.batch = torch.cat(batch_idx_list) if batch_idx_list else torch.empty(0, dtype=torch.long, device=device)
    return merge_batch


# ---------------- Main (Isotropic DiffAlign) ----------------

class DiffAlign(nn.Module):
    """
    Isotropic Gaussian Diffusion (v-parameterization; T steps)
    - Backbone: EGNN + CrossGraphAligner (only query coordinates move)
    - Output: v_t in merged (Q,R) order
    - Loss: v MSE + x0 anchor + optional repulsion
    """
    def __init__(
        self,
        node_feature_dim: int = 64,
        time_embed_dim: int = 32,
        query_embed_dim: int = 32,
        edge_encoder_dim: int = 64,
        gnn_hidden_dim: int = 128,
        gnn_layers_intra: int = 12,
        gnn_layers_intra_2: int = 4,
        gnn_layers_inter: int = 8,
        max_atom_types: int = 100,

        # Diffusion
        num_timesteps: int = 32,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule_type: str = 'cosine',

        # Repulsion
        repulsion_weight: float = 1e-2,
        repulsion_margin: float = 1.2,
        repulsion_exclude_hops: int = 3,
    ):
        super().__init__()

        # ---- Diffusion buffers (isotropic) ----
        self.num_timesteps = int(num_timesteps)
        if schedule_type == 'linear':
            betas = linear_beta_schedule(self.num_timesteps, beta_start, beta_end)
        elif schedule_type == 'cosine':
            betas = cosine_beta_schedule(self.num_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {schedule_type}")

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas', torch.sqrt(alphas))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('posterior_variance', betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod + 1e-12))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod + 1e-12))
        self.register_buffer('posterior_mean_coef2', torch.sqrt(alphas) * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod + 1e-12))

        # ---- Encoders ----
        self.edge_encoder = MLPEdgeEncoder(edge_encoder_dim, "relu")
        self.edge_encoder2 = MLPEdgeEncoder(edge_encoder_dim, "relu")

        self.node_encoder = nn.Sequential(
            nn.Embedding(max_atom_types, node_feature_dim),
            nn.SiLU(),
            nn.Linear(node_feature_dim, node_feature_dim),
        )
        self.time_encoder = DDPMTimeEncoder(time_embed_dim, activation=nn.SiLU)
        self.query_encoder = nn.Sequential(
            nn.Embedding(2, query_embed_dim),  # 0=ref, 1=query
            nn.SiLU(),
            nn.Linear(query_embed_dim, query_embed_dim),
        )

        gnn_in_node_dim = node_feature_dim + time_embed_dim + query_embed_dim

        self.intra_encoder = EGNN(
            in_node_nf=gnn_in_node_dim, in_edge_nf=edge_encoder_dim, hidden_nf=gnn_hidden_dim,
            n_layers=gnn_layers_intra, attention=True
        )
        self.cross_aligner = CrossGraphAligner(
            dim=gnn_hidden_dim,
            heads=4,
            dropout=0.1,
            coord_update=True,
            num_layers=gnn_layers_inter,
            recompute_each=1,
        )
        self.intra_encoder_2 = EGNN(
            in_node_nf=gnn_hidden_dim, in_edge_nf=edge_encoder_dim, hidden_nf=gnn_hidden_dim,
            n_layers=gnn_layers_intra_2, attention=True
        )

        # Repulsion hyperparams
        self.repulsion_weight = float(repulsion_weight)
        self.repulsion_margin = float(repulsion_margin)
        self.repulsion_exclude_hops = int(repulsion_exclude_hops)

    # -------------- Forward: predict v_t --------------

    def forward(self, query_batch: Batch, reference_batch: Batch, t: torch.Tensor,
                condition: bool = True) -> torch.Tensor:
        """
        Predict v_t for merged (query, reference) batches.
          - query_batch, reference_batch: torch_geometric.data.Batch
          - t: [G] timesteps (0..T-1) per graph
        """
        merged_batch = merge_graphs_in_batch(query_batch, reference_batch)
        if merged_batch.num_nodes == 0:
            device_to_use = query_batch.pos.device if hasattr(query_batch, 'pos') else 'cpu'
            return torch.zeros((0, 3), device=device_to_use)

        device = merged_batch.pos.device
        x_in = merged_batch.pos

        # (A) Query mask: only query coordinates are updated
        qmask_bool = ((merged_batch.graph_idx % 2) == 0)
        coord_mask = qmask_bool.float().unsqueeze(-1)

        # (B) Embeddings
        node_feat = self.node_encoder(merged_batch.atom_type)
        t_nodes = t[merged_batch.batch]  # expand per-graph timestep to nodes
        time_emb = self.time_encoder(t_nodes)
        is_query = ((merged_batch.graph_idx % 2) == 0).long()  # 1=query, 0=ref
        query_emb = self.query_encoder(is_query)
        h = torch.cat([node_feat, time_emb, query_emb], dim=-1)

        # (C) Intra-graph (stage 1): update only query coords
        edge_index, edge_type = extend_graph_order_radius(
            num_nodes=merged_batch.atom_type.size(0),
            pos=x_in,
            edge_index=merged_batch.edge_index,
            edge_type=merged_batch.edge_type,
            batch=merged_batch.graph_idx,
            order=3,
            cutoff=8,
            extend_order=True,
            extend_radius=True,
        )
        edge_length = get_distance(x_in, edge_index).unsqueeze(-1)
        e = self.edge_encoder(edge_length=edge_length, edge_type=edge_type)

        h, x = self.intra_encoder(
            h=h, x=x_in, edges=edge_index, edge_attr=e,
            coord_mask=coord_mask,
        )

        # (D) Cross (query moves)
        if condition:
            h, x = self.cross_aligner(h, x, batch=merged_batch.batch, graph_idx=merged_batch.graph_idx)

        # (E) Intra-graph (stage 2)
        edge_index2, edge_type2 = extend_graph_order_radius(
            num_nodes=merged_batch.atom_type.size(0),
            pos=x,
            edge_index=merged_batch.edge_index,
            edge_type=merged_batch.edge_type,
            batch=merged_batch.graph_idx,
            order=3,
            cutoff=8,
            extend_order=True,
            extend_radius=True,
        )
        edge_length2 = get_distance(x, edge_index2).unsqueeze(-1)
        e2 = self.edge_encoder2(edge_length=edge_length2, edge_type=edge_type2)

        h, x = self.intra_encoder_2(
            h=h, x=x, edges=edge_index2, edge_attr=e2,
            coord_mask=coord_mask,
        )

        # (F) Output: v = x - x_in
        v_hat = x - x_in
        return v_hat

    # -------------- Samplers --------------

    @torch.no_grad()
    def DDPM_Sampling_UFF(
        self,
        query_batch: Batch,
        reference_batch: Batch,
        *,
        clamp: float = 1e-10,
        cfg_scale: float = 1.0,
        # ---- UFF(PyTorch) options ----
        query_mols=None,                      # RDKit Mol list; None disables UFF
        pocket_mols=None,                     # RDKit Mol list; None disables pocket guidance
        uff_guidance_scale: float = 0.0,      # 0 disables UFF
        uff_inner_steps: int = 8,             # inner gradient steps for UFF
        uff_clamp: float = 1.0,               # clamp magnitude of forces
        uff_start_ratio: float = 0.0,         # skip UFF if (t/(T-1)) < ratio
        snr_gate_gamma: float = 1.0,          # gate(t) = (1 - σ_t)^γ
        # ---- Temperature option ----
        noise_temperature: float = 0.3,       # posterior noise temperature τ
        # ---- UFF dynamic nonbonded params ----
        uff_vdw_multiplier: float = 10.0,     # dynamic cutoff multiplier
        debug_log: bool = False,
    ):
        """
        Standard DDPM (v-param) + UFFTorch-based UFF steering on x0.
        Note: _refresh_nonbond_candidates is called with ligand+pocket coords merged.
        """

        device = self.betas.device
        qb = query_batch.to(device)
        rb = reference_batch.to(device)
        T = self.num_timesteps

        if qb.num_nodes == 0:
            return (torch.zeros((0, 3), device=device), None)

        # ---------- UFFTorch setup ----------
        use_uff = (
            uff_guidance_scale > 0.0
            and (query_mols is not None)
            and (pocket_mols is not None)
        )

        if use_uff:
            assert len(query_mols) == qb.num_graphs
            assert len(pocket_mols) == qb.num_graphs

            q_inputs_ref = build_uff_inputs(
                query_mols,
                device=device,
                dtype=torch.float32,
                vdw_distance_multiplier=uff_vdw_multiplier,
                ignore_interfragment_interactions=False,
            )
            p_inputs_ref = build_uff_inputs(
                pocket_mols,
                device=device,
                dtype=torch.float32,
                vdw_distance_multiplier=uff_vdw_multiplier,
                ignore_interfragment_interactions=False,
            )

            qp_inputs = merge_uff_inputs(
                q_inputs_ref,
                p_inputs_ref,
                ignore_interfragment_interactions=False,
                vdw_distance_multiplier=float(uff_vdw_multiplier),
            )

            uff_model = UFFTorch(qp_inputs).to(device).eval()
            uff_model._vdw_distance_multiplier = float(uff_vdw_multiplier)

            # 3. Query node indices
            mol_slices_q = [
                (qb.batch == i).nonzero(as_tuple=True)[0]
                for i in range(qb.num_graphs)
            ]
            gather_idx_q = torch.cat(mol_slices_q, dim=0).to(device)

            B = qb.num_graphs
            Nq = mol_slices_q[0].numel()
            for sl in mol_slices_q:
                assert sl.numel() == Nq, "Ligand atom counts must match across the batch."

            # 4. Freeze pocket coordinates (tensor conversion)
            def _mol_to_coords_tensor(m):
                conf = m.GetConformer()
                return torch.tensor(
                    [[conf.GetAtomPosition(k).x, conf.GetAtomPosition(k).y, conf.GetAtomPosition(k).z]
                     for k in range(m.GetNumAtoms())],
                    device=device, dtype=torch.float32
                )

            pocket_coords_fixed = torch.stack(
                [_mol_to_coords_tensor(m) for m in pocket_mols], dim=0
            )  # [B, Np, 3]

        else:
            uff_model = None

        # ---------- x_T init ----------
        x_t = torch.randn(
            (qb.num_nodes, 3),
            device=device,
            dtype=torch.float32,
        )

        # ---------- Reverse diffusion loop ----------
        for t in reversed(range(T)):
            t_graph = torch.full((qb.num_graphs,), t, device=device, dtype=torch.long)

            # Current coords + model prediction
            cur_q = qb.clone()
            cur_q.pos = x_t

            if cfg_scale == 1.0:
                v_hat_merged = self(cur_q, rb, t_graph, condition=True)
            else:
                v_u = self(cur_q, rb, t_graph, condition=False)
                v_c = self(cur_q, rb, t_graph, condition=True)
                v_hat_merged = v_u + cfg_scale * (v_c - v_u)

            merged = merge_graphs_in_batch(cur_q, rb, device=device)
            qmask = (merged.graph_idx % 2 == 0)
            v_hat = v_hat_merged[qmask]

            abar_t = self.sqrt_alphas_cumprod[t]
            sig_t  = self.sqrt_one_minus_alphas_cumprod[t]

            # 2) x0 prediction (reconstruction)
            x0_pred = abar_t * x_t - sig_t * v_hat

            # SNR Gate
            sigma_t_val = float(sig_t.item())
            h_t = 1.0 - sigma_t_val
            h_t = max(0.0, min(1.0, h_t))
            gate_t = h_t ** float(snr_gate_gamma)

            # 3) UFF Guidance: x0_pred → x0_star
            if (
                use_uff
                and (t / max(1, T - 1)) >= uff_start_ratio
                and gate_t > 0.0
            ):
                # (1) Extract ligand coords [B, Nq, 3]
                coords_q0 = x0_pred.index_select(0, gather_idx_q)
                coords_q0 = coords_q0.view(B, Nq, 3).detach()

                # (2) Build full coords (ligand+pocket) for neighbor refresh
                coords_full = torch.cat([coords_q0, pocket_coords_fixed], dim=1)

                with torch.no_grad():
                    uff_model._refresh_nonbond_candidates(coords_full)

                # (3) Gradient descent loop
                coords_q = coords_q0.clone()
                inner = max(1, int(uff_inner_steps))
                step_scale = (uff_guidance_scale * gate_t) / float(inner)
                
                with torch.enable_grad():
                    for _ in range(inner):
                        coords_q_req = coords_q.clone().requires_grad_(True)
                        
                        # Concatenate for energy; ligand needs grads, pocket stays fixed
                        coords_cat = torch.cat([coords_q_req, pocket_coords_fixed], dim=1)

                        E_b = uff_model(coords_cat)
                        
                        if not isinstance(E_b, torch.Tensor):
                            E_b = torch.as_tensor(E_b, device=device, dtype=coords_cat.dtype)
                        if E_b.ndim == 0:
                            E_b = E_b.unsqueeze(0)

                        grad_q, = autograd.grad(E_b.sum(), coords_q_req, create_graph=False)
                        
                        forces_q = (-grad_q).detach().clamp_(-uff_clamp, uff_clamp)
                        coords_q = (coords_q + step_scale * forces_q).detach()

                # (4) Apply refined coords to x0_star
                x0_star = x0_pred.clone()
                x0_star.index_copy_(0, gather_idx_q, coords_q.reshape(B * Nq, 3))

                if debug_log:
                    f_norm = forces_q.norm(dim=-1)
                    print(f"[UFF] t={t:02d} | Force Mean={f_norm.mean():.3f}")

            else:
                x0_star = x0_pred

            # 4) Posterior Mean (Standard DDPM)
            # mu_t = c1 * x0_star + c2 * x_t
            c1_t = self.posterior_mean_coef1[t]
            c2_t = self.posterior_mean_coef2[t]
            x_mean = c1_t * x0_star + c2_t * x_t

            # 5) Noise Addition
            if t > 0:
                var_t = self.posterior_variance[t]
                if noise_temperature != 1.0:
                    var_t = (noise_temperature ** 2) * var_t
                var_t = max(float(var_t), 1e-20)
                
                noise = torch.randn_like(x_t)
                x_t = x_mean + math.sqrt(var_t) * noise
            else:
                x_t = x_mean

        return (x_t, None)
