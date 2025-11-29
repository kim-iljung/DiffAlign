import torch
from torch import nn
import torch.nn.functional as F

def unsorted_segment_sum(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    """TensorFlow-style unsorted_segment_sum for 2D tensors [E, C]."""
    result = data.new_zeros((num_segments, data.size(1)))
    scatter_index = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, scatter_index, data)
    return result

class E_GCL(nn.Module):
    def __init__(
        self,
        input_nf: int,
        output_nf: int,
        hidden_nf: int,
        edges_in_d: int = 0,
        nodes_att_dim: int = 0,
        act_fn = nn.SiLU(),
        attention: bool = False,
        norm_diff: bool = True,
        tanh: bool = False,
        coords_range: float = 1.0,
        norm_constant: float = 0.0,
    ):
        super().__init__()
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        self.norm_constant = norm_constant

        edge_coords_nf = 1  # ||x_i - x_j||^2

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_nf * 2 + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

        coord_head = nn.Linear(hidden_nf, 1, bias=False)
        nn.init.xavier_uniform_(coord_head.weight, gain=1e-3)

        coord_mlp = [
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            coord_head,
        ]
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = coords_range
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def coord2radial(self, edge_index: torch.Tensor, coord: torch.Tensor):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]                         # [E,3]
        radial = torch.sum(coord_diff ** 2, dim=1, keepdim=True)     # [E,1]
        if self.norm_diff:
            norm = torch.sqrt(radial + 1e-8)
            coord_diff = coord_diff / (norm + self.norm_constant)
        return radial, coord_diff

    def edge_model(self, source, target, radial, edge_attr, edge_mask):
        if edge_attr is None:
            m_in = torch.cat([source, target, radial], dim=1)
        else:
            m_in = torch.cat([source, target, radial, edge_attr], dim=1)
        m = self.edge_mlp(m_in)
        if self.attention:
            m = m * self.att_mlp(m)
        if edge_mask is not None:
            m = m * edge_mask
        return m

    def node_model(self, x, edge_index, edge_feat, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_feat, row, num_segments=x.size(0))  # [N, H]
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        x_out = x + self.node_mlp(agg)
        return x_out

    def coord_model(self, coord, edge_index, coord_diff, edge_feat, edge_mask, coord_mask):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)  # [E,3] · [E,1] -> [E,3]
        if self.tanh:
            trans = trans * self.coords_range
        if edge_mask is not None:
            trans = trans * edge_mask
        dx = unsorted_segment_sum(trans, row, num_segments=coord.size(0))   # [N,3]

        if coord_mask is None:
            coord = coord + dx
        else:
            mask = coord_mask.unsqueeze(-1) if coord_mask.dim() == 1 else coord_mask
            coord = coord + dx * mask
        return coord

    def forward(
        self,
        h: torch.Tensor,               # [N, C_h]
        edge_index: torch.Tensor,      # [2, E]
        coord: torch.Tensor,           # [N, 3]
        edge_attr: torch.Tensor = None,# [E, C_e]
        node_attr: torch.Tensor = None,# [N, C_n]
        node_mask: torch.Tensor = None,# [N, 1] or [N]
        edge_mask: torch.Tensor = None,# [E, 1] or [E]
        coord_mask: torch.Tensor = None# [N, 1] or [N]; 1=update, 0=freeze
    ):
        radial, coord_diff = self.coord2radial(edge_index, coord)
        row, col = edge_index

        # Edge/message
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr, edge_mask)

        # Coordinate update (masked)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat, edge_mask, coord_mask)

        # Node update
        h = self.node_model(h, edge_index, edge_feat, node_attr)

        if node_mask is not None:
            h = h * node_mask
            coord = coord * node_mask

        return h, coord, edge_feat  # edge_feat is optional output

class EGNN(nn.Module):
    def __init__(
        self,
        in_node_nf: int,
        in_edge_nf: int,
        hidden_nf: int,
        device: str = 'cpu',
        act_fn = nn.SiLU(),
        n_layers: int = 4,
        attention: bool = False,
        norm_diff: bool = True,
        out_node_nf: int = None,
        tanh: bool = False,
        coords_range: float = 15.0,
        agg: str = 'sum',
        norm_constant: float = 0.0,
    ):
        super().__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range) / max(1, n_layers)
        if agg == 'mean':
            self.coords_range_layer = self.coords_range_layer * 19

        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.embedding_out = nn.Linear(hidden_nf, out_node_nf)

        for i in range(n_layers):
            self.add_module(
                f"gcl_{i}",
                E_GCL(
                    input_nf=hidden_nf,
                    output_nf=hidden_nf,
                    hidden_nf=hidden_nf,
                    edges_in_d=in_edge_nf,
                    act_fn=act_fn,
                    attention=attention,
                    norm_diff=norm_diff,
                    tanh=tanh,
                    coords_range=self.coords_range_layer,
                    norm_constant=norm_constant,
                ),
            )

        self.to(self.device)

    def forward(
        self,
        h: torch.Tensor,                  # [N, C_in]
        x: torch.Tensor,                  # [N, 3]
        edges: torch.Tensor,              # [2, E]
        edge_attr: torch.Tensor = None,   # [E, C_e]
        node_mask: torch.Tensor = None,   # [N, 1] or [N]
        edge_mask: torch.Tensor = None,   # [E, 1] or [E]
        coord_mask: torch.Tensor = None,  # [N, 1] or [N]; 1=update, 0=freeze
        node_attr: torch.Tensor = None,   # [N, C_n]
    ):
        h = self.embedding(h)
        for i in range(self.n_layers):
            h, x, _ = self._modules[f"gcl_{i}"](
                h, edges, x,
                edge_attr=edge_attr,
                node_attr=node_attr,
                node_mask=node_mask,
                edge_mask=edge_mask,
                coord_mask=coord_mask,
            )
        h = self.embedding_out(h)
        if node_mask is not None:
            h = h * node_mask
        return h, x
