import torch
import MinkowskiEngine as ME

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
def A_occupied_by_B(
    A: ME.SparseTensor,
    B: ME.SparseTensor | ME.CoordinateMapKey,
):
    cm = A.coordinate_manager

    if isinstance(B, ME.SparseTensor):
        strided_B_key = cm.stride(B.coordinate_map_key, A.tensor_stride)
    elif isinstance(B, ME.CoordinateMapKey):
        strided_B_key = cm.stride(B, A.tensor_stride)
    else:
        msg = "B must be either a SparseTensor or CoordinateMapKey."
        raise ValueError(msg)

    mask = torch.zeros(len(A), dtype=torch.bool, device=A.device)

    if cm.size(strided_B_key) == 0:
        return mask

    # only the exact match (kernel_size=1) is needed to determine occupancy
    kg = get_cube_kernel_generator(kernel_size=1)
    a_idx, _ = _sparse_tensor_key_map(
        A.coordinate_map_key,
        strided_B_key,
        kg,
        cm,
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

    assert A.tensor_stride == B.tensor_stride, "tensor_stride mismatch"

    out = ME.SparseTensor(
        features=torch.cat([A.F, B.F], dim=0),
        coordinates=torch.cat([A.C, B.C], dim=0),
        tensor_stride=A.tensor_stride,
        coordinate_manager=A.coordinate_manager,
    )
    return out
