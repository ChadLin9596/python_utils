import numpy as np
import open3d as o3d
import pptk
import pptk.kdtree

from . import utils_img


def _estimate_normals_by_pca(xyz, k=-1, r=10):
    return pptk.estimate_normals(xyz, k=k, r=r)


def _remove_mesh_vertices_by_masks(mesh, mask):

    vertices = mesh["vertices"]
    triangles = mesh["triangles"]
    vertex_normals = mesh["vertex_normals"]

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(vertex_normals)
    o3d_mesh.remove_vertices_by_mask(mask)

    R = {
        "vertices": np.asarray(o3d_mesh.vertices),
        "triangles": np.asarray(o3d_mesh.triangles),
        "vertex_normals": np.asarray(o3d_mesh.vertex_normals),
    }
    return R


def create_mesh_from_point_cloud_poisson(
    xyz,
    # estimate normals
    k=-1,
    r=10,
    # create_from_point_cloud_poisson
    depth=8,
    width=0,
    scale=1.1,
    linear_fit=False,
    # distance filtering
    dist_threshold=0.2,
    return_details=False,
):
    """
    ref:
    https://www.open3d.org/docs/release/python_api/open3d.geometry.TriangleMesh.html
    """

    xyz = xyz.astype(np.float64)
    normals = _estimate_normals_by_pca(xyz, k=k, r=r)
    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(xyz)  # d: double
    o3d_pc.normals = o3d.utility.Vector3dVector(normals)
    o3d_pc.orient_normals_to_align_with_direction(np.r_[0, 0, 1])

    # construct mesh
    func = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson
    o3d_mesh, densities = func(
        o3d_pc,
        depth=depth,
        width=width,
        scale=scale,
        linear_fit=linear_fit,
    )
    vertices = np.asarray(o3d_mesh.vertices)
    triangles = np.asarray(o3d_mesh.triangles)
    vertex_normals = np.asarray(o3d_mesh.vertex_normals)
    densities = np.asanyarray(densities)
    raw_details = {
        "vertices": vertices,
        "triangles": triangles,
        "vertex_normals": vertex_normals,
        "densities": densities,
    }

    # remove vertices by distances
    _tree = pptk.kdtree._build(xyz)
    _indices = pptk.kdtree._query(_tree, raw_details["vertices"], k=1)
    _indices = np.hstack(_indices)
    _dists = np.sqrt(np.sum((vertices - xyz[_indices]) ** 2, axis=-1))

    R = _remove_mesh_vertices_by_masks(raw_details, _dists > dist_threshold)
    if return_details:
        return R, raw_details
    return R


def mesh_to_point_map(
    vertices,
    triangles,
    intrinsic,
    extrinsic,
    H,
    W,
    invalid_value=np.nan,
    return_details=False,
):
    """
    ref:
    https://www.open3d.org/docs/release/python_api/open3d.t.geometry.RaycastingScene.html
    """

    rc_scene = o3d.t.geometry.RaycastingScene()
    rc_scene.add_triangles(
        vertex_positions=o3d.core.Tensor(vertices.astype(np.float32)),
        triangle_indices=o3d.core.Tensor(triangles.astype(np.uint32)),
    )

    # (H, W, 6)
    _rays = rc_scene.create_rays_pinhole(
        intrinsic_matrix=o3d.core.Tensor(intrinsic),
        extrinsic_matrix=o3d.core.Tensor(extrinsic),
        width_px=W,
        height_px=H,
    )

    # Compute the ray intersections.
    ans = rc_scene.cast_rays(_rays)

    rays = _rays.numpy()
    ray_origins = rays[..., :3]
    ray_directs = rays[..., 3:]

    # fmt: off
    ans = {
        # ray distance to the first hit
        "t_hit": ans["t_hit"].numpy(),  # (H, W)

        # ids
        "geometry_ids": ans["geometry_ids"].numpy(),  # (H, W)
        "primitive_ids": ans["primitive_ids"].numpy(),  # (H, W)

        # barycentric coordinates of the hit points within the hit triangles
        "primitive_uvs": ans["primitive_uvs"].numpy(),  # (H, W, 2)

        # normals of the hit triangles
        "primitive_normals": ans["primitive_normals"].numpy(),
    }
    # fmt: on

    M = np.isfinite(ans["t_hit"])

    point_map = np.full((H, W, 3), invalid_value, dtype=np.float32)
    point_map[M] = ray_origins[M] + ans["t_hit"][M][:, None] * ray_directs[M]

    if return_details:
        details = {}
        details.update(ans)
        details.update(
            {
                "ray_origins": ray_origins,
                "ray_directs": ray_directs,
            }
        )
        return point_map, details

    return point_map


def meshes_to_point_map(
    list_of_vertices,
    list_of_triangles,
    intrinsic,
    extrinsic,
    H,
    W,
    invalid_value=np.nan,
    return_details=False,
):

    rc_scene = o3d.t.geometry.RaycastingScene()
    for vertices, triangles in zip(list_of_vertices, list_of_triangles):
        rc_scene.add_triangles(
            vertex_positions=o3d.core.Tensor(vertices.astype(np.float32)),
            triangle_indices=o3d.core.Tensor(triangles.astype(np.uint32)),
        )

    # (H, W, 6)
    _rays = rc_scene.create_rays_pinhole(
        intrinsic_matrix=o3d.core.Tensor(intrinsic),
        extrinsic_matrix=o3d.core.Tensor(extrinsic),
        width_px=W,
        height_px=H,
    )

    # Compute the ray intersections.
    ans = rc_scene.cast_rays(_rays)

    rays = _rays.numpy()
    ray_origins = rays[..., :3]
    ray_directs = rays[..., 3:]

    # fmt: off
    ans = {
        # ray distance to the first hit
        "t_hit": ans["t_hit"].numpy(),  # (H, W)

        # ids
        "geometry_ids": ans["geometry_ids"].numpy(),  # (H, W)
        "primitive_ids": ans["primitive_ids"].numpy(),  # (H, W)

        # barycentric coordinates of the hit points within the hit triangles
        "primitive_uvs": ans["primitive_uvs"].numpy(),  # (H, W, 2)

        # normals of the hit triangles
        "primitive_normals": ans["primitive_normals"].numpy(),
    }
    # fmt: on

    M = np.isfinite(ans["t_hit"])

    point_map = np.full((H, W, 3), invalid_value, dtype=np.float32)
    point_map[M] = ray_origins[M] + ans["t_hit"][M][:, None] * ray_directs[M]

    if return_details:
        details = {}
        details.update(ans)
        details.update(
            {
                "ray_origins": ray_origins,
                "ray_directs": ray_directs,
            }
        )
        return point_map, details

    return point_map


def mesh_to_depth_map(
    vertices,
    triangles,
    intrinsic,
    extrinsic,
    H,
    W,
    invalid_value=np.inf,
):

    point_map = mesh_to_point_map(
        vertices,
        triangles,
        intrinsic,
        extrinsic,
        H,
        W,
        invalid_value=np.nan,
    )

    M = ~np.any(np.isnan(point_map), axis=-1)

    cam_xyz = utils_img._trans_from_camera_to_world(point_map[M], extrinsic)
    depth_map = np.full((H, W), fill_value=invalid_value, dtype=np.float64)
    depth_map[M] = cam_xyz[:, 2]
    return depth_map


def meshes_to_depth_map(
    list_of_vertices,
    list_of_triangles,
    intrinsic,
    extrinsic,
    H,
    W,
    invalid_value=np.inf,
):
    point_map = meshes_to_point_map(
        list_of_vertices,
        list_of_triangles,
        intrinsic,
        extrinsic,
        H,
        W,
        invalid_value=np.nan,
    )

    M = ~np.any(np.isnan(point_map), axis=-1)

    cam_xyz = utils_img._trans_from_camera_to_world(point_map[M], extrinsic)
    depth_map = np.full((H, W), fill_value=invalid_value, dtype=np.float64)
    depth_map[M] = cam_xyz[:, 2]
    return depth_map


def labels_from_cluster_dbscan(xyz, eps=0.3, min_points=10):
    """
    https://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html#open3d.geometry.PointCloud.cluster_dbscan
    """

    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(xyz)
    labels = o3d_pc.cluster_dbscan(
        eps=eps,
        min_points=min_points,
        print_progress=False,
    )
    labels = np.asarray(labels)
    return labels
