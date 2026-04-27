import math
import torch
import torch.nn as nn
import MinkowskiEngine as ME
from . import utils_torch

####################
# KERNEL GENERATOR #
####################


def get_cube_kernel_generator(kernel_size, stride=1, dilation=1, dimension=3):
    """for kernel_size = 3, the kernel region is a 3x3x3 cube"""

    return ME.KernelGenerator(
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        region_type=ME.RegionType.HYPER_CUBE,
        dimension=dimension,
    )


def get_cross_kernel_generator(kernel_size, stride=1, dilation=1, dimension=3):
    """for kernel_size = 3, the kernel region is a 3x3x3 cross"""

    return ME.KernelGenerator(
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        region_type=ME.RegionType.HYPER_CROSS,
        dimension=dimension,
    )


########################
# NEIGHBORHOOD MAPPING #
########################


@torch.no_grad()
def _sparse_tensor_key_map(
    A: ME.CoordinateMapKey,
    B: ME.CoordinateMapKey,
    kernel_generator: ME.KernelGenerator,
    coordinate_manager: ME.CoordinateManager,
    device="cuda",
):

    kg = kernel_generator
    km = coordinate_manager.kernel_map(
        A,
        B,
        kernel_size=kg.kernel_size,
        stride=kg.kernel_stride,
        dilation=kg.kernel_dilation,
        region_type=kg.region_type,
        region_offset=kg.region_offsets,
    )

    a_keys, b_keys = [], []
    for _, pair in km.items():
        a, b = pair
        a_keys.append(a.long())
        b_keys.append(b.long())

    if len(a_keys) == 0 or len(b_keys) == 0:
        a_keys = torch.empty(0, dtype=torch.long, device=device)
        b_keys = torch.empty(0, dtype=torch.long, device=device)
    else:
        a_keys = torch.cat(a_keys)
        b_keys = torch.cat(b_keys)

    return a_keys, b_keys


@torch.no_grad()
def sparse_tensor_map(
    A: ME.SparseTensor,
    B: ME.SparseTensor,
    kernel_generator=get_cube_kernel_generator(1),
):

    if A.coordinate_manager is not B.coordinate_manager:
        raise ValueError("A and B must share the same coordinate_manager.")

    # shorthanded
    cm = A.coordinate_manager
    ak = A.coordinate_map_key
    bk = B.coordinate_map_key
    kg = kernel_generator

    exp_stride = [b // a for a, b in zip(A.tensor_stride, B.tensor_stride)]
    ker_stride = list(kg.kernel_stride)

    if ker_stride != exp_stride:
        msg = f"kernel_generator stride {ker_stride} does not match: "
        msg += f"A.tensor_stride {A.tensor_stride} "
        msg += f"B.tensor_stride {B.tensor_stride} "
        msg += f"expected stride {exp_stride})."
        raise ValueError(msg)

    return _sparse_tensor_key_map(ak, bk, kg, cm, device=A.device)


@torch.no_grad()
def _A_occupied_by_B(
    A: ME.CoordinateMapKey,
    B: ME.CoordinateMapKey,
    coordinate_manager: ME.CoordinateManager,
    device="cuda",
):

    # only the exact match (kernel_size=1) is needed to determine occupancy
    kg = get_cube_kernel_generator(kernel_size=1)
    a_idx, _ = _sparse_tensor_key_map(
        A,
        B,
        kg,
        coordinate_manager,
        device=device,
    )
    return a_idx


@torch.no_grad()
def A_occupied_by_B(
    A: ME.SparseTensor,
    B: ME.SparseTensor,
):

    if A.coordinate_manager is not B.coordinate_manager:
        raise ValueError("A and B must share the same coordinate_manager.")

    if A.tensor_stride != B.tensor_stride:
        raise ValueError("A and B must have the same tensor_stride.")

    mask = torch.zeros(len(A), dtype=torch.bool, device=A.device)
    a_idx = _A_occupied_by_B(
        A.coordinate_map_key,
        B.coordinate_map_key,
        A.coordinate_manager,
        device=A.device,
    )
    mask[a_idx] = True

    return mask


@torch.no_grad()
def A_occupied_by_B_key(
    A: ME.SparseTensor,
    B: ME.CoordinateMapKey,
):
    mask = torch.zeros(len(A), dtype=torch.bool, device=A.device)
    a_idx = _A_occupied_by_B(
        A.coordinate_map_key,
        B,
        A.coordinate_manager,
        device=A.device,
    )
    mask[a_idx] = True

    return mask


##################
# SET OPERATIONS #
##################


@torch.no_grad()
def set_difference(A: ME.SparseTensor, B: ME.SparseTensor):
    """A - B"""

    assert A.tensor_stride == B.tensor_stride, "tensor_stride mismatch"

    occupied = A_occupied_by_B(A, B)
    if not torch.any(occupied):
        return A

    if torch.all(occupied):
        return None  # A - B is empty; avoid constructing 0-voxel SparseTensor

    keep = ~occupied

    out = ME.SparseTensor(
        features=A.F[keep],
        coordinates=A.C[keep],
        tensor_stride=A.tensor_stride,
        coordinate_manager=A.coordinate_manager,
    )
    return out


@torch.no_grad()
def set_disjoint_union(A: ME.SparseTensor, B: ME.SparseTensor):
    """A U B, assume A and B don't have intersection"""

    if B is None:
        return A
    if A is None:
        return B

    assert A.tensor_stride == B.tensor_stride, "tensor_stride mismatch"

    if len(B) == 0:
        return A
    if len(A) == 0:
        return B

    out = ME.SparseTensor(
        features=torch.cat([A.F, B.F], dim=0),
        coordinates=torch.cat([A.C, B.C], dim=0),
        tensor_stride=A.tensor_stride,
        coordinate_manager=A.coordinate_manager,
    )
    return out


@torch.no_grad()
def append_unique_coords(A, B) -> ME.SparseTensor:

    C = set_difference(B, A)
    R = set_disjoint_union(A, C)
    return R


###############
# TRANSFORMER #
###############


def scaled_dot_product_attention(Q, K, V, Q_idx, K_idx, n_queries):
    """
    Q:     (Na, H, dh)
    K:     (Nb, H, dh)
    V:     (Nb, H, dv)
    Q_idx: (E,) int64, edge indices into Q (query side)
    K_idx: (E,) int64, edge indices into K/V (key/value side)
    returns: (Na, H, dv)
    """
    E = Q_idx.shape[0]
    H = Q.shape[1]
    D = V.shape[-1]

    logits = (Q[Q_idx] * K[K_idx]).sum(-1) / math.sqrt(Q.shape[-1])  # (E, H)
    logits = logits.reshape(-1)  # (EH,)

    h_idx = torch.arange(H, device=Q.device)

    # (E, 1) * scalar + (1, H) -> (E, H)
    groups = Q_idx[:, None] * H + h_idx
    groups = groups.reshape(-1)  # (EH,)

    weights = utils_torch.segmented_softmax(logits, groups, n_queries * H)
    weights = weights.reshape(E, H)  # (E, H)

    out = torch.zeros((n_queries, H, D), device=Q.device, dtype=Q.dtype)
    out = torch.scatter_reduce(
        input=out,
        dim=0,
        # (E,) -> (E, 1, 1) -> (E, H, D)
        index=Q_idx[:, None, None].expand(-1, H, D),
        # (E, H) -> (E, H, 1) * (E, H, D)
        src=weights[..., None] * V[K_idx],
        reduce="sum",
        include_self=True,
    )
    return out


class MultiHeadAttention(nn.Module):

    def __init__(self, ca: int, cb: int, d: int, n_heads=1):
        super().__init__()
        assert d % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d // n_heads

        self.q = nn.Linear(ca, d, bias=False)
        self.k = nn.Linear(cb, d, bias=False)
        self.v = nn.Linear(cb, d, bias=False)
        self.proj = nn.Linear(d, ca, bias=False)

    def forward(self, Fa, Fb, a_idx, b_idx):

        Na = Fa.shape[0]
        H = self.n_heads
        d_H = self.head_dim

        Q = self.q(Fa).reshape(Na, H, d_H)  # (Na, H, dh)
        K = self.k(Fb).reshape(-1, H, d_H)  # (Nb, H, dh)
        V = self.v(Fb).reshape(-1, H, d_H)  # (Nb, H, dh)

        # out: (Na, H, dH)
        out = scaled_dot_product_attention(Q, K, V, a_idx, b_idx, Na)
        out = out.reshape(Na, H * d_H)

        return Fa + self.proj(out)


class NeighborhoodCrossAttention(MultiHeadAttention):

    def __init__(self, ca, cb, d, n_heads=1, kernel_size=3):

        super().__init__(ca, cb, d, n_heads)
        self.kernel_gen = get_cube_kernel_generator(kernel_size)

    def forward(self, A, B):

        a_idx, b_idx = sparse_tensor_map(A, B, self.kernel_gen)
        if a_idx.numel() == 0:
            return A.F

        return super().forward(A.F, B.F, a_idx, b_idx)


class FullCrossAttention(MultiHeadAttention):

    def forward(self, A, B):

        Fa, Fb = A.F, B.F
        Na, Nb = Fa.shape[0], Fb.shape[0]
        dev = Fa.device

        # [0, 0, ... 1, 1, .., Na-1, Na-1]
        Q_idx = torch.arange(Na, device=dev).repeat_interleave(Nb)
        # [0, 1, ..., Na-1, 0, 1, ...]
        K_idx = torch.arange(Nb, device=dev).repeat(Na)

        return super().forward(Fa, Fb, Q_idx, K_idx)
