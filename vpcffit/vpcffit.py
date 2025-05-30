#%%

#%%
import numpy as np
import atomap.api as am
import hyperspy.api as hs
from ase.visualize import view
from ase import Atoms
from abtem.atoms import orthogonalize_cell
from copy import copy
from pathlib import Path
from tqdm import tqdm
import ase.io as aio
from typing import Literal
from tkinter import messagebox
from scipy.fftpack import fft, ifft
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
from scipy.spatial.distance import squareform, pdist
from SingleOrigin import UnitCell, get_vpcf, detect_peaks
from networkx import draw_networkx, from_edgelist, connected_components
from .utils import suppress_stdout, xy_to_rt, rt_to_xy


def extract_peak_coords(origin: tuple[float, float],
                        v_pcf: np.ndarray,
                        method: Literal["max", "com", "gauss"] = "gauss",
                        thresh: float = 0.1,
                        min_dist: int = 4) -> np.ndarray:
    """Extracts peak coordinates from a vPCF via max filtering.
    Args:
        origin: The origin of the vPCF; peak coordinates will be transformed to be centered on the origin.
        v_pcf: The vPCF to extract peaks from.
        method: Which peak-finding method to use: maximum intensity ("max"), center of mass ("com") or gaussian
                fitting ("gauss").
        thresh: Minimum pixel value allowed to be considered a peak, passed to SingleOrigin.detect_peaks. Defualt is 0.
        min_dist: Minimum distance (px) allowed between peaks, passed to SingleOrigin.detect_peaks. Default is 4.

    Returns:
        An array containing (y, x, sig_y, sig_x, ellipticity)
    """
    if method == "max":
        peaks = np.argwhere(detect_peaks(v_pcf, min_dist=min_dist, thresh=thresh))
        return np.array([((peak[0] - origin[0]) * 0.01, (peak[1] - origin[1]) * 0.01, 0, 0, 1) for peak in peaks])
    elif method in ("gauss", "com"):
        sig_vPCF = hs.signals.Signal2D(v_pcf)
        positions = am.get_atom_positions(sig_vPCF, pca=False,  threshold_rel=thresh, separation=min_dist)
        sub = am.Sublattice(positions, image=sig_vPCF.data, fix_negative_values=True)
        sub.find_nearest_neighbors()
        sub.refine_atom_positions_using_center_of_mass(show_progressbar=False)
        if method == "gauss":
            sub.refine_atom_positions_using_2d_gaussian(show_progressbar=False)
        return np.array(((sub.y_position - origin[0]),
                         (sub.x_position - origin[1]),
                         sub.sigma_y, sub.sigma_x, sub.ellipticity)).T


def merge_close(coordinates: list[tuple[float, float]],
                tol: float,
                show_graph: bool = False)\
        -> list[tuple[float, float]]:
    """Merge closely-neighboring points into a single point, with coordinates at the center of mass of the neighborhood.
    Args:
        coordinates: A collection of cartesian point coordinates.
        tol: The maximum Euclidean distance to count as a neighbor.
        show_graph: Whether to plot the near-neighbor graph; useful for debugging. Default is False.
    Returns:
        The list of coordinates after merging.
    """
    # Algorithm: 1) Get the full distance matrix (fast for a small number of points)
    #            2) For each point, find all of its neighbors (within tol distance)
    #            3) Construct an undirected graph from the points (neighbors share an edge)
    #            4) Find all the connected components of the graph
    #            5) Add a new point at the center of mass of each connected component
    #            6) Remove the unmerged points
    dists = squareform(pdist(np.array(coordinates), metric="euclidean"))
    close_neighbor_edges = set()
    for i in range(len(coordinates)):
        neighbors = np.argwhere(dists[i, :] < tol).flatten()
        if neighbors.size == 1:  # Disregard points which only neighbor themselves
            continue
        # Set comprehension to ignore duplicate edges; frozenset is unordered and hashable
        new_edges = {frozenset([i, n]) for n in neighbors if n != i}
        close_neighbor_edges.update(new_edges)

    edgelist = list(map(tuple, close_neighbor_edges))  # Format as list of tuples for networkx to consume
    graph = from_edgelist(edgelist)
    if show_graph:
        draw_networkx(graph, node_size=10, arrowsize=20,
                      node_color="#ffffff", edge_color="#ffffff", font_color="#0088ff")
        messagebox.showinfo(title="networkx.draw(graph)",
                            message="Displaying point connectivity graph; press OK to continue.",
                            detail="Distances in connectivity graph are not real-space Euclidean distances.")

    conn_comp = list(connected_components(graph))
    merged_coords, drop_indices = [], []
    for comp in conn_comp:
        merge_x, merge_y = None, None
        for index in comp:
            drop_indices.append(index)
            x, y = coordinates[index]
            if merge_x is None:  # Initial values
                merge_x = x
                merge_y = y
            else:
                merge_x += x
                merge_y += y
        merge_x /= len(comp)
        merge_y /= len(comp)
        merged_coords.append((merge_x, merge_y))

    final_coords = copy(coordinates)
    for i in sorted(drop_indices, reverse=True):  # Remove unmerged points from end first
        del final_coords[i]
    final_coords.extend(merged_coords)
    return final_coords


def get_vpcf_from_cif2(cif_path: Path | str,
                      axes: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
                      e: str,
                      e_ignore: str | list[str] | None = None,
                      xlims: tuple[float, float] = (-10, 10),
                      ylims: tuple[float, float] = (-10, 10),
                      px_size: float = 0.01,
                      guass_sigma: float = 0,
                      vpcf_tol: float = 0.5)\
        -> np.ndarray:
    """Generate the ideal peak coordinates of a vPCF given a cif file and zone axis, in polar form.
    Args:
        cif_path: The path to the cif file (string or pathlib Path object).
        axes: A tuple containing the zone axis to project, as well as orthogonal basis vectors to complete
            the coordinate system.  The format is (zone axis, basis 2, basis 3), and each vector should be a
            tuple of three integers.  Example: ((0, 0, 1), (5, 0, 0), (0, 5, 0)).
        e: Symbol of the element of interest (e.g. "Al").
        e_ignore: Symbol of the element to ignore when generating the unit cell (e.g. "N"); can also be a
            list of symbols or None. Default is None.
        xlims: The x-limits for the vPCF. xlims[0] should be less than xlims[1], and the range must include 0.
            Default is (-10, 10).
        ylims: The y-limits for the vPCF. ylims[0] should be less than ylims[1], and the range must include 0.
            Default is (-10, 10).
        px_size: The pixel size of the vPCF, in the same units as the unit cell atom column coordinates.
            Default is 0.01.
        guass_sigma: The sigma for the gaussia filter user to blue the computed vPCF (px). Default is 0 (no blur).
            Default is 1.
        vpcf_tol: The tolerance for merging nearby peaks in the vPCF (A). Default is 0.5.
    Returns:
        The peaks of the vPCF calculated from the cif file along the given zone axis, in polar coordinates.
    """
    za, a1, a2 = axes
    uc = UnitCell(str(cif_path))  # Must cast path object to string for SingleOrigin to handle it
    # SingleOrigin expects ignore_elements to always be a list
    if type(e_ignore) is str:
        e_ignore = [e_ignore]
    elif e_ignore is None:
        e_ignore = []

    with suppress_stdout():
        uc.project_zone_axis(za, a1, a2,
                             ignore_elements=e_ignore,
                             reduce_proj_cell=False)
    vpcf, origin = get_vpcf(xlim=xlims, ylim=ylims, d=px_size,
                            coords1=uc.at_cols.loc[uc.at_cols["elem"] == e, ["x", "y"]].to_numpy())
    vpcf = gaussian_filter(vpcf, guass_sigma)
    # TODO: allow controlling this function's parameters
    peak_origin_coordinates = extract_peak_coords(origin, vpcf, method="com")
    # merged_coordinates = merge_close(peak_origin_coordinates, vpcf_tol)
    return peak_origin_coordinates


def get_vpcf_from_cif(cif_path: Path | str,
                      axes: tuple[int, int, int],
                      e: str,
                      xlims: tuple[float, float] = (-10, 10),
                      ylims: tuple[float, float] = (-10, 10),
                      px_size: float = 0.02,
                      guass_sigma: float = 0)\
        -> np.ndarray:
    """Generate the ideal peak coordinates of a vPCF given a cif file and zone axis, in polar form.
    Args:
        cif_path: The path to the cif file (string or pathlib Path object).
        axes: A tuple containing the zone axis to project.
        e: Symbol of the element of interest (e.g. "Al").
        xlims: The x-limits for the vPCF. xlims[0] should be less than xlims[1], and the range must include 0.
            Default is (-10, 10).
        ylims: The y-limits for the vPCF. ylims[0] should be less than ylims[1], and the range must include 0.
            Default is (-10, 10).
        px_size: The pixel size of the vPCF, in the same units as the unit cell atom column coordinates.
            Default is 0.01.
        guass_sigma: The sigma for the gaussia filter user to blue the computed vPCF (px). Default is 0 (no blur).
            Default is 1.
    Returns:
        The peaks of the vPCF calculated from the cif file along the given zone axis, in polar coordinates.
    """
    struct = aio.read(cif_path)
    # TODO: I NEED TO UPDATE THIS TO ADD IN-PLANE ROTATION FOR INDIVIDUAL ZA & CIF
    new_ZA = rotate_to_direction(struct, axes, 0)
    tilex = round( 50 / new_ZA.cell[0][0])
    tiley = round( 50 / new_ZA.cell[1][1])
    model = new_ZA*(tilex,tiley,1)
    unique_species = sorted(set(model.get_chemical_symbols()))
    coord_DB = {}
    for atom in unique_species:

        coord_DB[atom] = np.array([model.positions[i]
                                   for i, condition in enumerate(model.get_chemical_symbols())
                                   if condition==atom])[:,0:2]

    vpcf, origin = get_vpcf(xlim=xlims, ylim=ylims, d=px_size,coords1=coord_DB[e])
    vpcf = gaussian_filter(vpcf, guass_sigma)
    # TODO: allow controlling this function's parameters
    peak_origin_coordinates = extract_peak_coords(origin, vpcf, method="com")
    # merged_coordinates = merge_close(peak_origin_coordinates, vpcf_tol)
    return peak_origin_coordinates
#########################
# UNFINISHED CODE BELOW #
#########################


def update_dmax(good_fit_tol: float,
                mu: float,
                sig: float,
                prefactor: float = 20) -> float:
    """Update the maximum tolerable distance based on statistics about the distance distribution.
    Args:
        good_fit_tol: User-specified goodness-of-fit parameter.
        mu: The mean of the distance distribution.
        sig: The standard deviation of the distance distribution.
        prefactor: Multiplier for good_fit_tol used when the fit is bad. Default is 20 (see Zhang 1993).
    Returns:
        The new d_max value.
    """
    if mu < good_fit_tol:  # Good registration
        return mu + 3*sig
    elif mu < 3*good_fit_tol:  # Decent registration
        return mu + 2*sig
    elif mu < 6*good_fit_tol:  # Okay-ish registration
        return mu + sig
    else:  # mu >= 6*good_fit_tol
        # Bad fit; we need to resort to back-up d_max computation
        # Zhang 1993 does this by choosing xi to be in the valley to the right of the modal distance (for some
        # histogram binning). We have too few points to reasonably bin, so instead we fall back to an initial
        # guess of prefactor*good_fit_tol.
        return prefactor*good_fit_tol


def similarity_transform(polar_pt: tuple[float, float],
                         r_scale: float, rot: float) -> tuple:
    """Similarity (rotation & scaling) transform for a single point, in polar (r, t) coordinates."""
    return (polar_pt[0]*r_scale), (polar_pt[1]+rot) % (2*np.pi), *polar_pt[2:]


def loss(params: tuple[float, float],
         pts: list[tuple[float, float]],
         tree: KDTree,
         good_fit_tol: float,
         is_coarse: bool = False) -> float:
    """Loss function which can either dynamically update d_max (for fine fitting) or not (for coarse global fitting).
    Args:
        params: Tuple of the form (r_scale, rot), which are the fitting parameters to be optimized.
        pts: The list of points currently being fit (Cartesian).
        tree: A k-D tree representing the points currently being fit to (Cartesian).
        good_fit_tol: User-specified goodness-of-fit parameter, used to calculate d_max.
        is_coarse: Set to True to disable dynamic updates of d_max (for coarse search).

    Returns:
        The loss for the current fitting parameters.
    """
    try:
        exp_transformed = [rt_to_xy(similarity_transform(xy_to_rt(pt), *params)) for pt in pts]
        # We can't just use distance_upper_bound param of query, since it assigns d=inf to unmatched points
        ds, _ = tree.query(exp_transformed)
        if not is_coarse:  # Dynamically update d_max during iteration for fine fitting
            mu, sig = np.mean(ds), np.std(ds)
            d_max = update_dmax(good_fit_tol, mu, sig)
            ds = [d for d in ds if d <= d_max]

        if len(ds) == 0:
            return np.inf
        else:
            return sum(ds)

    except ValueError:
        # Some optimize methods (e.g. L-BFGS-B) seem to occasionally try (nan, nan) as fit parameters
        # I don't know why... but at least this catches that behavior
        print(f"NaN encountered in fit parameters: {params}")


def coarse_search(bounds: tuple[tuple[float, float], tuple[float, float]],
                  n_r: int, n_t: int,
                  pts: list[tuple[float, float]],
                  tree: KDTree,
                  good_fit_tol: float,
                  return_all: bool = False)\
        -> tuple[float, float, float] | tuple[float, float, float, list[tuple[float, float, float]]]:
    """Grid-based brute-force search over (scale, rotation) parameters, meant for coarse global optimization.
    Args:
        bounds: Tuples containing the bounds of the search. Scale bounds first, rotation bounds second.
        n_r: The number of search points to generate on the scale axis between the given bounds (inclusive).
        n_t: The number of search points to generate on the rotation axis between the given bounds (inclusive).
        pts: The set of points used to query the tree for the overall loss.
        tree: The tree to be queried for the overall loss.
        good_fit_tol: User-defined goodness-of-fit parameter.
        return_all: If True, returns a list of all checked (scale, rotation, loss) values, useful for mapping the
            optimization space. Default is False.

    Returns:
        The scale and rotaiton parameters associated with the minimum loss value, as well as that loss value. If
            return_all is True, additionally returns a list of all checked (scale, rotation, loss) points.
    """
    r_scale_span = np.linspace(bounds[0][0], bounds[0][1], n_r, endpoint=True)
    rot_span = np.linspace(bounds[1][0], bounds[1][1], n_t, endpoint=True)

    minimum, min_r, min_t = np.inf, 1, 0
    collector = []
    for r in r_scale_span:
        for t in rot_span:
            _l = loss((r, t), pts, tree, good_fit_tol, is_coarse=True)
            if return_all:
                collector.append((r, t, _l))
            if _l < minimum:
                minimum = _l
                min_r, min_t = r, t
    if not return_all:
        return min_r, min_t, minimum
    else:
        return min_r, min_t, minimum, collector


def binarize_atomic_columns(coords, px_size=0.1, probe=0.55):
    number_dist_x = int(np.round((coords[:, 0].max()+np.abs(coords[:, 0].min())) / px_size, 0)) + 1
    number_dist_y = int(np.round((coords[:, 1].max()+np.abs(coords[:, 1].min())) / px_size, 0)) + 1

    pc1 = np.empty((number_dist_x, number_dist_y))

    for index, pts in enumerate(coords):
        x_coor = int(np.round(pts[0] / px_size, 0))
        y_coor = int(np.round(pts[1] / px_size, 0))
        pc1[x_coor, y_coor] = 1

    return gaussian_filter(pc1, sigma = probe / px_size)


def crop_at_position(array, crop_height, crop_width, pos_x, pos_y):
    """ Crop images from the center to given height and width (px units)"""
    # Get the dimensions of the original array
    original_height, original_width = array.shape

    # Calculate the starting position for cropping
    start_height = max(0, pos_y - crop_height//2)
    start_width = max(0, pos_x - crop_width//2)
    final_height = min(original_height, start_height + crop_height)
    final_width = min(original_width, start_width + crop_width)

    return array[start_height:final_height, start_width:final_width]

def circle_mask(radius, tks):
    size = 2 * radius + 1
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 < radius**2
    unmask = x**2 + y**2 < (radius-tks)**2
    circle_array = np.zeros((size, size), dtype=int)
    circle_array[mask] = 1
    circle_array[unmask] = 0
    circle_array[0:size,radius] = 1
    circle_array[radius,0:size] = 1
    return circle_array


def add_mask(arr,arr_shape,center, mask_center, mask_indices, mask_flat):
    start_idx = np.array(center) - mask_center
    target_i = ((mask_indices[0] + start_idx[0]) % arr_shape[0]).flatten()
    target_j = ((mask_indices[1] + start_idx[1]) % arr_shape[1]).flatten()
    arr[target_i, target_j] += mask_flat

def prepare_add_mask(mask):
    mask_shape = mask.shape
    mask_center = np.array(mask_shape) // 2
    mask_indices = np.indices(mask_shape)
    mask_flat = mask.flatten()
    return mask_center, mask_indices, mask_flat

# Function to align point clouds by rotating and scaling using cross correlation
def align_point_clouds(pc1: np.ndarray,
                       pc2: np.ndarray,
                       r_limit: float = 500,
                       mask_size: int = 40,
                       radial_pts: int = 200,
                       angular_pts: int = 3600,
                       scale: float = 1,
                       plot = False) -> (np.ndarray, np.ndarray, float):
    """Align two point clouds by cross-correlation in theta.

    Args:
        pc1: Reference point cloud (Cartesian).
        pc2: Target point cloud (Cartesian).
        mask_size: Siz of the mask used for cross-correlation (px) used to expand the point of the point cloud.
        radial_pts: Number of pixels in the radial direction.
        angular_pts: Number of pixels in the angualr direction.
        scale: Radial scaling for the target point cloud.

    Returns: Target point cloud (rotated and scaled), reference point cloud (limited to points within the maximum
    radius of the target point cloud, if it is smaller), and the angle that was used to rotate the target point cloud.
    """
    pc1_polar = xy_to_rt(pc1)
    pc2_polar = xy_to_rt(pc2)
    px_size_t = 2*np.pi / angular_pts
    scale_t = pc2_polar[:, 0] * scale
    pc2_polar[:, 0] = scale_t

    px_size_r = r_limit / radial_pts
    pc1_polar = pc1_polar[pc1_polar[:, 0] <= r_limit]
    pc2_polar = pc2_polar[pc2_polar[:, 0] <= r_limit]

    pc1_bin = np.empty((radial_pts, angular_pts))
    pc2_bin = np.empty((radial_pts, angular_pts))
    mask = circle_mask(mask_size, 1)
    pre_mask = prepare_add_mask(mask)
    pc1_shape = pc1_bin.shape
    pc2_shape = pc2_bin.shape

    for index, pts in enumerate(pc1_polar):
        angle = int(np.round(pts[1]/px_size_t, decimals=0))
        r = int(np.round(pts[0]/px_size_r, decimals=0))
        add_mask(pc1_bin, pc1_shape,(r,angle), *pre_mask)
    for index, pts in enumerate(pc2_polar):
        angle = int(np.round(pts[1]/px_size_t, decimals=0))
        r = int(np.round(pts[0]/px_size_r, decimals=0))
        add_mask(pc2_bin, pc2_shape,(r,angle), *pre_mask)


    # Compute the FFT along the x-axis of both images
    fft_image1_x = fft(pc1_bin, axis=1)
    fft_image2_x = fft(pc2_bin, axis=1)

    # Compute the cross-correlation along the x-axis using the FFT results
    cross_corr_x = ifft(fft_image1_x * np.conj(fft_image2_x), axis=1).real
    # Sum over the y-axis to get a single cross-correlation vector along the x-axis
    cross_corr_x = np.sum(cross_corr_x, axis=0)
    # Shift the zero-frequency component to the center of the spectrum
    cross_corr_x = np.fft.fftshift(cross_corr_x)
    # Find the position of the maximum value in the cross-correlation
    max_index_x = np.argmax(cross_corr_x)
    # Compute the shift required to align the images along the x-axis
    shift_x = max_index_x - (pc1_bin.shape[1] // 2)
    # Transform to angular units
    rotation_angle = np.mod(shift_x * px_size_t, 2*np.pi)

    pc2_polar[:, 1] = np.mod(pc2_polar[:, 1] + rotation_angle, np.pi * 2)
    pc1_cart = rt_to_xy(pc1_polar)
    pc2_cart = rt_to_xy(pc2_polar)

    if plot:
        import matplotlib.pyplot as plt
        fig1 = plt.figure(figsize=(50, 50))
        plt.imshow(pc1_bin, cmap='Grays')
        plt.gca().invert_yaxis()
        fig2 = plt.figure(figsize=(50, 50))
        plt.imshow(pc2_bin, cmap='Grays')
        plt.gca().invert_yaxis()
        fig3 = plt.figure(figsize=(50, 50))
        plt.plot(cross_corr_x, label="Cross-correlation")
        #plt.show()
        return pc2_cart, pc1_cart, rotation_angle, [fig1,fig2,fig3]
    else:
        return pc2_cart, pc1_cart, rotation_angle


def near_neighbor_distance(target_pc: np.ndarray,
                           pc: np.ndarray,
                           nnd: Literal["ANND", "SNND"] = 'ANND',
                           **kwargs) -> tuple[tuple[np.ndarray, np.ndarray, float], float, float]:
    """ Find the sum of nearest neighbor distance between two point clouds making sure the two clouds are rotated to
    the best cross correlation.
    Args:
        target_pc: Target point cloud.
        pc: Other point cloud.
        nnd: NND metric (either "SNND" for the sum or "ANND" for the average. Default is "SNND".
        **kwargs: Additional keyword arguments will be passed to align_point_cloud; see its documentation for details.

    Returns: The return from align_point_cloud, the near neighbor distances (sum or average), and the propogated
    uncertainty.
    """

    if nnd not in ["ANND", "SNND"]:
        raise ValueError("nnd must be one of 'ANND' or 'SNND'")

    aligned_pc = align_point_clouds(pc, target_pc, **kwargs)
    reference_forest = KDTree(aligned_pc[0][:, 0:2])
    ds1, ds1_index = reference_forest.query(aligned_pc[1][:, 0:2])
    # Uncertainties for the first tree
    matches1 = aligned_pc[0][ds1_index]
    vects1 = aligned_pc[1][:, 0:2]-matches1[:, 0:2]
    vects1 = vects1[(vects1[:,0] != 0) & (vects1[:,1] != 0)]
    sigs_1 = []
    for i, v in enumerate(vects1):
        sigs_1.append(np.sqrt((v[0]*np.sqrt(aligned_pc[1][i, 2]**2 + matches1[i, 2]**2))**2 +
                              (v[1]*np.sqrt(aligned_pc[1][i, 3]**2 + matches1[i, 3]**2))**2) / np.linalg.norm(v))
    sig_overall_1 = np.sqrt(np.sum([s**2 for s in sigs_1]))

    # reference_forest2 = KDTree(aligned_pc[1][:, 0:2])
    # ds2, ds2_index = reference_forest2.query(aligned_pc[0][:, 0:2])
    # # Uncertainties for the second tree
    # matches2 = aligned_pc[1][ds2_index]
    # vects2 = aligned_pc[0][:, 0:2]-matches2[:, 0:2]
    #
    # vects2 = vects2[(vects2[:,0] != 0) & (vects2[:,1] != 0)]
    # sigs_2 = []
    # for i, v in enumerate(vects2):
    #     sigs_2.append(np.sqrt((v[0]*np.sqrt(aligned_pc[0][i, 2]**2 + matches2[i, 2]**2))**2 +
    #                           (v[1]*np.sqrt(aligned_pc[0][i, 3]**2 + matches2[i, 3]**2))**2) / np.linalg.norm(v))
    # sig_overall_2 = np.sqrt(np.sum([s**2 for s in sigs_2]))

    if 'scale' in kwargs.keys():
        kwargs['scale'] = 1/kwargs['scale']
    aligned_pc_reverse = align_point_clouds(target_pc, pc, **kwargs)
    reference_forest2 = KDTree(aligned_pc_reverse[0][:, 0:2])
    ds2, ds2_index = reference_forest2.query(aligned_pc_reverse[1][:, 0:2])
    # Uncertainties for the second tree
    matches2 = aligned_pc_reverse[0][ds2_index]
    vects2 = aligned_pc_reverse[1][:, 0:2]-matches2[:, 0:2]
    vects2 = vects2[(vects2[:,0] != 0) & (vects2[:,1] != 0)]
    sigs_2 = []
    for i, v in enumerate(vects2):
        sigs_2.append(np.sqrt((v[0]*np.sqrt(aligned_pc_reverse[1][i, 2]**2 + matches2[i, 2]**2))**2 +
                              (v[1]*np.sqrt(aligned_pc_reverse[1][i, 3]**2 + matches2[i, 3]**2))**2) / np.linalg.norm(v))
    sig_overall_2 = np.sqrt(np.sum([s**2 for s in sigs_2]))


    if nnd == 'SNND':
        near_neighbor_distances = (sum(ds1) + sum(ds2))
        sig_total = np.sqrt(sig_overall_1**2 + sig_overall_2**2)  # Overall uncertainty in SNND
    else:
        if len(ds1) and len(ds2) != 0:
            near_neighbor_distances = sum(ds1)/len(ds1) + sum(ds2)/len(ds2)
            sig_total = np.sqrt((sig_overall_1/len(ds1))**2 + (sig_overall_2/len(ds2))**2)

        else:
            raise ArithmeticError("Divide by zero: no matching points found!")
    #print(sum(ds1)/len(ds1) ,sum(ds2)/len(ds2))
    return aligned_pc, near_neighbor_distances, sig_total


def rotate_to_direction(structure, axis, angle=0):
    # Convert direction to a catersian vector
    direction = np.dot(structure.cell.T, np.array(axis))

    # Do not rotate if the Zone axis coincide with the c-axis in cartesian coordinates
    if np.array_equal(direction/np.linalg.norm(direction), [0, 0, 1]):
        atoms = structure.copy()
        atoms.rotate(angle, 'z', rotate_cell=True)
        return orthogonalize_cell(atoms)
    elif np.array_equal(direction/np.linalg.norm(direction), [0, 0, -1]):
        atoms = structure.copy()
        atoms.rotate(180, 'x', rotate_cell=True)
        atoms.rotate(angle, 'z', rotate_cell=True)
        return orthogonalize_cell(atoms)

    # Define rotational matrix based on the axis to rotate around and angle
    z_axis = [0, 0, 1]
    rotation_axis = np.cross(z_axis, direction).astype(np.float64)
    rotation_axis /= np.linalg.norm(rotation_axis)
    rotation_angle = np.arccos(np.dot(z_axis, direction)/np.linalg.norm(direction))
    rotation_matrix = rotation_matrix_axis_angle(rotation_axis, rotation_angle)

    # Rotate the lattice vectors
    rotated_cell = np.dot(structure.cell, rotation_matrix)
    # Rotate the atomic positions
    rotated_positions = np.dot(structure.get_positions(), rotation_matrix)
    # Create a new rotated structure with updated lattice vectors and atomic positions
    rotated_structure = Atoms(cell=rotated_cell, positions=rotated_positions, numbers=structure.get_atomic_numbers(),
                              pbc=True)
    rotated_structure.rotate(angle, 'z', rotate_cell=True)

    return orthogonalize_cell(rotated_structure)


def rotation_matrix_axis_angle(axis, theta):
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2)
    b, c, d = -axis * np.sin(theta / 2)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rotation_matrix = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                                [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                                [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
    return rotation_matrix


def supercell_axis_cif(list_cif,
                       axes,
                       size = 100,  # A
                       in_plane_rotation = 0,
                       view_structure = False):
    struct = aio.read(list_cif)
    if view_structure:
        view(struct)

    # TODO: I NEED TO UPDATE THIS TO ADD IN-PLANE ROTATION FOR INDIVIDUAL ZA & CIF
    new_ZA = rotate_to_direction(struct, axes, in_plane_rotation)
    tilex = round( size / new_ZA.cell[0][0])
    tiley = round( size / new_ZA.cell[1][1])
    struc_DB = new_ZA*(tilex,tiley,1)

    return struc_DB


def get_image_from_atoms(model, atomic_specie = 'Hf', probe_size = 0.55, px_size = 0.1, im_size_pix = 512):
    unique_species = sorted(set(model.get_chemical_symbols()))
    coord_DB = {}
    for atom in unique_species:
        coord_DB[atom] = np.array([model.positions[i]
                                   for i, condition in enumerate(model.get_chemical_symbols())
                                   if condition==atom])[:,0:2]

    img = binarize_atomic_columns(coord_DB[atomic_specie], px_size=px_size, probe=probe_size)
    if img.shape[0]>im_size_pix or img.shape[1]>im_size_pix:
        calculated_images = [crop_at_position(img, im_size_pix, im_size_pix, img.shape[0]//2 , img.shape[0]//2),
                                    px_size]
    else:
        pad_x = (im_size_pix - img.shape[0]) // 2  # Padding for rows
        pad_y = (im_size_pix - img.shape[1]) // 2  # Padding for columns
        calculated_images = [np.pad(img, ((pad_x, im_size_pix - img.shape[0] - pad_x),
                                              (pad_y, im_size_pix - img.shape[1] - pad_y)),
                                        mode='constant', constant_values=0),px_size]
    return calculated_images


def create_vpcfs_images(image: tuple[np.ndarray, float],
                        vpcf_pixel_size=0.02,
                        fit_atom_gaussian=True,
                        blur_vpcf=3,
                        separation=8,
                        xlims: tuple[float, float] = (-10, 10),
                        ylims: tuple[float, float] = (-10, 10),
                        show_progressbar: bool = False,
                        **kwargs) -> tuple[tuple,np.ndarray, am.Sublattice ]:
    """Generate vPCFs from images.

    Args:
        dict_images: A tuple (image, pixel size)}.
        vpcf_pixel_size: The pixel size of the generated vPCFs (Angstrom).
        fit_atom_gaussian: If True, refine atom column positions by fitting gaussians, else uses center-of-mass
        fitting only. Default is True.
        blur_vpcf: The sigma for a gaussian blur filter to be applied to the generated vPCFs (pixels).
        separation: Minimum allowed separation between atom columns during fitting (pixels).
        xlims: The x-limits for the vPCF. xlims[0] should be less than xlims[1], and the range must include 0.
            Default is (-10, 10).
        ylims: The y-limits for the vPCF. ylims[0] should be less than ylims[1], and the range must include 0.
            Default is (-10, 10).
        **kwargs: Keyword arguments to be passed to extract_peak_coords.

    Returns: Three dictionaries containg respectively: vPCFs as images (with origin offsets), vPCF peak coordinates,
    and atomap Sublattice objects for each image.
    """

    img = hs.signals.Signal2D(image[0])
    image_fit = am.get_atom_positions(img, pca=True, separation=separation)
    sublattices = am.Sublattice(image_fit, image=img.data)
    sublattices.find_nearest_neighbors()
    sublattices.refine_atom_positions_using_center_of_mass(show_progressbar=show_progressbar)
    if fit_atom_gaussian:
        sublattices.refine_atom_positions_using_2d_gaussian(show_progressbar=show_progressbar)
    coords = sublattices.atom_positions * image[1]

    vpcfs, origin = get_vpcf(xlim=xlims,
                             ylim=ylims,
                             coords1=coords,
                             d=vpcf_pixel_size)

    blur_vpcfs = gaussian_filter(vpcfs, sigma = blur_vpcf)

    vpcfs_peaks = extract_peak_coords(origin, blur_vpcfs, **kwargs)


    return (blur_vpcfs, origin), vpcfs_peaks, sublattices
