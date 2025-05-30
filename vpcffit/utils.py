import sys
import json
import tkinter
import warnings
import numpy as np
from os import devnull
from pathlib import Path
from numpy.typing import ArrayLike
from scipy.spatial import ConvexHull
from matplotlib import pyplot as plt
from contextlib import contextmanager
from tkinter.filedialog import askopenfilename, askopenfilenames, asksaveasfilename
from itertools import product
import ase.io as aio
from math import gcd
from functools import reduce


def select_files(single: bool = False, **kwargs) -> str | tuple[str, ...]:
    """Helper function for selecting file(s) with a Tkinter dialog which forces the window to be focused on creation.

    Args:
        single: If True, calls askopenfilename. If False, calls askopenfilenames.
        **kwargs: Passed to tkinter.filedialog.askopenfilename(s) (depending on if single is True or False).

    Returns:
        The filename or list of filenames.
    """
    root = tkinter.Tk()
    root.iconify()
    root.attributes('-topmost', True)
    root.update()

    if "parent" in kwargs:
        raise ValueError("Do not manually set 'parent'")

    try:
        if single is True:
            return askopenfilename(parent=root, **kwargs)
        else:
            return askopenfilenames(parent=root, **kwargs)
    finally:
        root.attributes('-topmost', False)
        root.destroy()


def xy_to_rt(xy: np.ndarray) -> np.ndarray:
    """Convert an array of points from Cartesian (x, y) to polar (r, t) coordinates."""
    r = np.hypot(xy[:, 0], xy[:, 1])
    t = np.mod(np.arctan2(xy[:, 1], xy[:, 0]), 2*np.pi)
    return np.column_stack((r, t, xy[:, 2:]))


def rt_to_xy(rt: np.ndarray) -> np.ndarray:
    """Convert an array of points from polar (r, t) to Cartesian (x, y) coordinates."""
    x = rt[:, 0] * np.cos(rt[:, 1])
    y = rt[:, 0] * np.sin(rt[:, 1])
    return np.column_stack((x, y, rt[:, 2:]))


@contextmanager
def suppress_stdout():
    """Context manager to suppress printing to stdout by piping into devnull; by Dave Smith:
    https://thesmithfam.org/blog/2012/10/25/temporarily-suppress-console-output-in-python/"""
    with open(devnull, "w") as dn:
        old_stdout = sys.stdout
        sys.stdout = dn
        try:
            yield
        finally:
            sys.stdout = old_stdout


def _json_ndarray_convert(x):
    """Function by AKX on StackOverflow: https://stackoverflow.com/a/65354261/1867985"""
    if hasattr(x, "tolist"):  # numpy arrays have this
        return {"$array": x.tolist()}  # Make a tagged object
    raise TypeError(x)


def _json_ndarray_deconvert(x):
    """Function by AKX on StackOverflow: https://stackoverflow.com/a/65354261/1867985"""
    # Doing it this way makes the de-conversion recursive, so nested lists are de-converted to arrays correctly
    if len(x) == 1:  # Might be a tagged object
        key, value = next(iter(x.items()))  # Grab the tag and value
        if key == "$array":  # If the tag is correct cast back to array
            return np.array(value)
    return x


def save_vpcfs(vpcfs: dict[str, np.ndarray],
               fname: str = "", directory: str | Path = ".", **kwargs) -> None:
    """ Save a dictionary containing calculated vPCFs. If both fname and directory are specified, no user interaction
    is required.

    Args:
        vpcfs: The dictionary of vPCFs to be saved to disk.
        fname: The filename to use when saving the vPCFs.
        directory: The directory in which to save the vPCFs (passed to pathlib).
        **kwargs: Passed to tkinter.filedialog.asksaveasfilename (if user interaction is needed).

    """
    if fname != "" and directory != ".":  # Both directory and filename pre-specified, no user interaction needed
        saveloc = Path(directory) / Path(fname).with_suffix(".json")
    else:
        root = tkinter.Tk()
        root.iconify()
        root.attributes('-topmost', True)
        root.update()
        try:
            saveloc = asksaveasfilename(initialdir=directory, initialfile=fname, parent=root,
                                        defaultextension=".json", filetypes=[("JSON Files", ".json")], **kwargs)
        finally:
            root.attributes('-topmost', False)
            root.destroy()

    with open(saveloc, "w") as f:
        json.dump(vpcfs, f, default=_json_ndarray_convert)


def load_vpcfs() -> dict[str, list[tuple[float, float]]]:
    with open(select_files(single=True, filetypes=[("JSON Files", ".json")]), "r") as f:
        return json.load(f, object_hook=_json_ndarray_deconvert)


def gui_crop(uncropped: ArrayLike,
             square: bool = False,
             bias: str = "tl",
             cmap: str = "bone",
             crop_timeout: int = 300,
             conf_timeout: int = 60,
             dpi: int = 200) -> np.ndarray:
    """Quickly crop an image to a square using matplotlib by specifying four corners within which
    to crop.

    Parameters
    ----------
    uncropped : ArrayLike
        The uncropped image, as a numpy array or some format which can be cast to a numpy array.
    square : bool, optional
        Whether to return a rectangular (False) or square (True) region. Defult is False (rectangular).
    bias : str, optional
        A string containing exactly one of ["t", "b"] and exactly one of ["l", "r"] which defines
        the bias direction for cropping (which sides of a non-square region we prefer to keep when
        cropping down to a square).  Default is "tl".
    cmap : str, optional
        The colormap to be passed to matplotlib.  See the matplotlib documentation at
        https://matplotlib.org/stable/tutorials/colors/colormaps.html for available options.
        Default is "bone".
    crop_timeout : int, optional
        How long to wait to get user input for the crop before giving up.  If cropping
        times out, the uncropped image will be returned. Default is 300 seconds (5 minutes).
        Set to -1 to disable timing out.
    conf_timeout : int, optional
        How long to wait for the user to confirm their crop before giving up.  If confirmation
        times out, cropping will be retried.  Default is 60 seconds (1 minute).  Set to -1
        to disable timing out.
    dpi : int, optional
        The DPI with which to show images for cropping and confirmation. Default is 200;
        the default matplotlib setting is 100.  Increasing this value can make precision cropping
        easier, while decreasing this value is better for low-resolution displays.

    Raises
    ------
    RuntimeWarning
        Raised if the user manually cancels cropping or if the cropping times out.
    UserWarning
        May be raised when attempting to crop from extremely non-square regions; matplotlib will
        also render a blank canvas.  Usually solvable by retrying the crop with a more square
        region.
    ValueError
        This error will be raised if an invalid bias string is passed.

    Returns
    -------
    ArrayLike
        Returns a numpy array representing an image.  If the crop is successful, the cropped image
        will be returned.  If the user manually cancels the crop or if cropping times out, the
        uncropped image will be returned and a RuntimeWarning will be raised.
    """

    # Crop-confirm loop
    while True:
        # Plot and show starting image
        with plt.rc_context({"figure.dpi": dpi}):
            fig, ax = plt.subplots()
            plt.axis("off")
            instructions = "Click on the four corners within which to crop\n" +\
                           "Right-click (or press Backspace/Delete) to remove last point\n" +\
                           "Middle-click or pressing Enter to cancel the crop"
            fig.text(0.5, 0.9, instructions, horizontalalignment="center", size="xx-small")
            ax.imshow(uncropped, cmap=cmap)
            # Get the user to click the four corners
            corner_points = plt.ginput(n=4, timeout=crop_timeout)
            plt.close(fig)

        # Sanity check and handle input errors
        if len(set(corner_points)) != 4:
            if len(corner_points) == 4:
                # User probably unintentionally clicked the same point twice
                warnings.warn("Identical points in input, please try again", RuntimeWarning)
                continue

            # Intentional exit or timeout: return uncropped
            warnings.warn("Cropping terminated early, returning uncropped image",
                          RuntimeWarning)
            return np.asarray(uncropped)

        # If all is well, proceed with cropping
        rect = _minimal_rect(corner_points, bias=bias, square=square)

        minx, maxx = rect[0][0], rect[1][0]
        miny, maxy = rect[0][1], rect[3][1]

        cropped = np.asarray(uncropped)[miny:maxy, minx:maxx]

        # Show the cropped image and confirm or retry
        with plt.rc_context({"figure.dpi": dpi}):
            fig, ax = plt.subplots()
            plt.axis("off")
            instructions = "To retry the crop click any mouse button\n" +\
                           "Otherwise to confrim hit any keyboard button"
            fig.text(0.5, 0.9, instructions, horizontalalignment="center", size="xx-small")
            ax.imshow(cropped, cmap=cmap)
            # Wait for the user to click  to confirm
            if plt.waitforbuttonpress(timeout=conf_timeout):
                plt.close(fig)
                return cropped

            plt.close(fig)
            continue


def _squarify(points: list[tuple[float, float]], bias: str):
    """Make a rectangular region into a square region.

    Parameters
    ----------
    points : List[Tuple[float, float]]
        List of points as tuples; must be ordered clockwise, starting from the top left.  These
        must define a rectangle (e.g. points[0][0] == points[3][0], points[0][1] == points[1][1],
        etc.) else we'll get nonsense.
    bias : str
        A string containing exactly one of ["t", "b"] and exactly one of ["l", "r"] which defines
        the bias direction (which sides we prefer to keep).

    Returns
    -------
    List[Tuple[flost, float]]
        The list of points, cropped down to be a square.

    """
    # The square must have a side length equal to the smallest
    vlen = points[3][1] - points[0][1]  # Left side length == right side length
    hlen = points[1][0] - points[0][0]  # Top side length == bottom side length

    # Make the points mutable
    _points = [list(point) for point in points]

    if hlen < vlen:
        if "t" in bias:
            # Crop off the bottom
            _points[2][1] = points[1][1] + hlen
            _points[3][1] = points[0][1] + hlen
        elif "b" in bias:
            # Crop off the top
            _points[0][1] = points[3][1] - hlen
            _points[1][1] = points[2][1] - hlen
    else:  # vlen < hlen
        if "l" in bias:
            # Crop off the right
            _points[1][0] = points[0][0] + vlen
            _points[2][0] = points[3][0] + vlen
        if "r" in bias:
            # Crop off the left
            _points[0][0] = points[1][0] - vlen
            _points[3][0] = points[2][0] - vlen

    # Turn the points back into immutable tuples
    return [tuple(point) for point in _points]


def _interior_round(points: list[tuple[float, float]]):
    """Round floats to integers, ensuring that rounding occurs toward the interior of a rectangular region.

    Parameters
    ----------
    points : List[Tuple[float, float]]
        List of points as (x, y) tuples, where xs and ys are floats; must be ordered clockwise,
        starting from the top left.

    Returns
    -------
    List[Tuple[int, int]]
        The list of points, rounded to integers in the interior of the above region.

    """
    # Make the data structure mutable
    _points = [list(point) for point in points]
    tl, tr, br, bl = _points[0], _points[1], _points[2], _points[3]

    # X - round up
    tl[0] = int(np.ceil(tl[0]))
    bl[0] = int(np.ceil(bl[0]))
    # X - round down
    tr[0] = int(np.floor(tr[0]))
    br[0] = int(np.floor(tr[0]))
    # Y - round up
    tl[1] = int(np.ceil(tl[1]))
    tr[1] = int(np.ceil(tr[1]))
    # Y - round down
    bl[1] = int(np.floor(bl[1]))
    br[1] = int(np.floor(br[1]))

    # Turn the points back into immutable tuples
    return [tuple(point) for point in [tl, tr, br, bl]]


def _minimal_rect(points: list[tuple[float, float]], bias: str = "tl", square: bool = False):
    """
    Parameters
    ----------
    points : List[Tuple[float, float]]
        A list of points, as returned from matplotlib.pyplot.ginput.  Note that if these points
        descirbe a region too different from a rectangle, this funcion MAY SILENTLY FAIL.  Be
        careful with such inputs.
    bias : str, optional
        A string containing exactly one of ["t", "b"] and exactly one of ["l", "r"] which defines
        the bias direction (which sides we prefer to keep). Default is "tl".
    square : bool, optional
        If False, a rectangular region will be returned. If True, the rectangular region will be further
        cropped to be square. Defualt is False.

    Raises
    ------
    NotImplementedError
        Non-quadrilateral regions are not yet implemented, so this error will be raised if the list
        of distinct points is not exactly length 4.
    ValueError
        This error will be raised if an invalid bias string is passed, or if the given points do
        not form a convex region.
    RuntimeError
        Rasied by failed sanity checking; may be raised if the passed points are highly non-square.

    Returns
    -------
    List[Tuple[int, int]]
        A list of points describing the minimal square crop solution.  The points will be sorted in
        clockwise order, starting from the top-leftmost point, and will be integer-valued.

    """
    # List of tuple is the default returned from matplotlib.pyplot.ginput,
    #  but in our case we want to make sure we have exactly 4 points.  They
    #  should also be distinct
    if len(set(points)) != 4:
        raise NotImplementedError("Exactly four (distinct) points must be passed: "
                                  "current implementation only supports quadrilateral regions.")

    # Bias string must contain exactly one of ["t", "b"], and exactly one of ["l", "r"]
    if ["t" in bias, "b" in bias].count(True) != 1:
        raise ValueError("Invalidly formatted bias string: "
                         "bias string must contain exactly one of 't' or 'b'.")
    if ["l" in bias, "r" in bias].count(True) != 1:
        raise ValueError("Invalidly formatted bias string: "
                         "bias string must contain exactly one of 'l' or 'r'.")

    # Form convex hull from points to ensure convexity
    hull = ConvexHull(np.asarray(points), incremental=True)
    verts = hull.vertices
    sorted_points = [list(points[i]) for i in verts]
    hull_vertices = [list(hull.points[i]) for i in verts]
    if not sorted_points == hull_vertices:
        raise ValueError("Passed points do not form a convex region: "
                         "minimal rectangular region is undefined on concave regions.")

    sorted_x, sorted_y = sorted([tup[0] for tup in points]), sorted([tup[1] for tup in points])
    # The following is only guaranteed to work for convex quadrilateral regions
    minimal_rect = [(sorted_x[1], sorted_y[1]),
                    (sorted_x[-2], sorted_y[1]),
                    (sorted_x[-2], sorted_y[-2]),
                    (sorted_x[1], sorted_y[-2])]
    if square:
        cropped = _squarify(minimal_rect, bias)
        # Convert indices to integers for slicing
        cropped = _interior_round(cropped)
    else:
        cropped = _interior_round(minimal_rect)

    # Sanity check: the rectangle should be entirely inside the hull, so adding the it's
    #  points to the hull should not change the vertex list
    hull.add_points(cropped)
    if not np.all(hull.vertices == verts):
        raise RuntimeError("Sanity check failed: cropped falls outside convex hull! "
                           "Was your region highly non-rectangular?")
    return cropped

def normalize(vector):
    """ Normalize the vector by dividing by the GCD of its non-zero elements. """
    non_zero_elements = vector[vector != 0]
    if len(non_zero_elements) == 0:
        return vector  # Return the zero vector as is
    vector = np.array(vector, dtype=int)
    divisor = reduce(gcd, non_zero_elements)
    return vector // np.abs(divisor)

def create_zone_axes(max_index: int,
                     cif: str) -> list[np.ndarray]:
    """Create a list of unique zone axes in the form of [[h,k,l]] for the space group in the cif file.

    Args:
        max_index: integer for the maximum index to generate zone axes.
        cif: a string with the path to load a cif file. Make sure the correct space group is specified in the cif file.

    Returns:
        List of zone axes.
    """
    vectors = np.array([np.array(v) for v in product(range(-max_index,max_index + 1), repeat=3)])
    zone_axes = [v for v in vectors if not np.all(v == [0, 0, 0])]
    atoms = aio.read(cif)
    space_group = atoms.info.get("spacegroup")
    rotations = space_group.rotations
    unique_zone_axes = []
    for zone_axis in reversed(zone_axes):
        equivalent_zone_axes = []
        flag = True
        for sym_op in rotations:
            rotated_zone_axis = np.dot(sym_op, zone_axis)
            if any(np.allclose(rotated_zone_axis, existing_axis) for existing_axis in unique_zone_axes):
                flag = False
                break
            if not any(np.allclose(rotated_zone_axis, existing_axis) for existing_axis in equivalent_zone_axes):
                equivalent_zone_axes.append(rotated_zone_axis)
        for a in unique_zone_axes:
            nor_array = normalize(np.array(a))
            if any(np.allclose(a, axis) for axis in equivalent_zone_axes):
                flag = False
            if any(np.allclose(nor_array, axis) for axis in equivalent_zone_axes):
                flag = False
        if flag:
            if not any(np.allclose(zone_axis, existing_axis) for existing_axis in unique_zone_axes):
                if not any(np.allclose(normalize(zone_axis), existing_axis) for existing_axis in unique_zone_axes):
                    unique_zone_axes.append(zone_axis)

    print(str(len(unique_zone_axes))+" unique zone axes for space group: "+space_group.symbol)
    return [normalize(np.array(vector)) for vector in unique_zone_axes]

