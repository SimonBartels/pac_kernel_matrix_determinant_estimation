# bug fix for clean_figure
import numpy as np
import tikzplotlib
from tikzplotlib._cleanfigure import _corners2D, _segments_intersect, _isempty, _diff


def _move_points_closer_fix(xLim, yLim, data):
    return data

def _remove_NaNs_fix(data):
    """Removes superflous NaNs in the data, i.e. those at the end/beginning of the data and consecutive ones.

    :param linehandle: matplotlib linehandle object

    :returns: data without NaNs
    """
    id_nan = np.any(np.isnan(data), axis=1)
    id_remove = np.argwhere(id_nan).reshape((-1,))
    if _isempty(id_remove):
        pass
    else:
        id_remove = id_remove[
            np.concatenate(
                [_diff(id_remove, axis=0) == 1, np.array([False]).reshape((-1,))]
            )
        ]

    #id_first = np.argwhere(np.logical_not(id_nan))[0]
    #id_last = np.argwhere(np.logical_not(id_nan))[-1]
    id_first = np.argwhere(np.logical_not(id_nan))
    id_last = np.argwhere(np.logical_not(id_nan))

    if _isempty(id_first):
        # remove entire data
        id_remove = np.arange(len(data))
    else:
        id_remove = np.concatenate(
            [np.arange(0, id_first[0]), id_remove, np.arange(id_last[-1] + 1, len(data))]
        )
    data = np.delete(data, id_remove, axis=0)
    return data


def _segment_visible_fix(data, dataIsInBox, xLim, yLim):
    """Given a bounding box {x,y}Lim, determine whether the line between all
    pairs of subsequent data points [data(idx,:)<-->data(idx+1,:)] is visible.
    There are two possible cases:
    1: One of the data points is within the limits
    2: The line segments between the datapoints crosses the bounding box

    :param data: array of data points. Shape [N, 2]
    :type data: np.ndarray
    :param dataIsInBox: boolen mask that specifies if data point lies within visual box
    :type dataIxInBox: np.ndarray
    :param xLim: x axes limits
    :type xLim: list, np.array
    :param yLim: y axes limits
    :type yLim: list, np.array

    :returns : mask
    """
    n = np.shape(data)[0]
    mask = np.zeros(n - 1) == 1

    # Only check if there is more than 1 point
    if n > 1:
        # Define the vectors of data points for the segments X1--X2
        idx = np.arange(n - 1)
        X1 = data[idx, :]
        X2 = data[idx + 1, :]

        # One of the neighbors is inside the box and the other is finite
        thisVisible = np.logical_and(dataIsInBox[idx], np.all(np.isfinite(X2), 1))
        nextVisible = np.logical_and(dataIsInBox[idx + 1], np.all(np.isfinite(X1), 1))

        bottomLeft, topLeft, bottomRight, topRight = _corners2D(xLim, yLim)

        left = _segments_intersect(X1, X2, bottomLeft, topLeft)
        right = _segments_intersect(X1, X2, bottomRight, topRight)
        bottom = _segments_intersect(X1, X2, bottomLeft, bottomRight)
        top = _segments_intersect(X1, X2, topLeft, topRight)

        # Check the result
        mask1 = np.logical_or(thisVisible, nextVisible)
        mask2 = np.logical_or(left, right)
        mask3 = np.logical_or(top, bottom)

        mask = np.logical_or(mask1, mask2)
        mask = np.logical_or(mask3, mask)

    return mask


def fix_tikzplotlib_clean_figure():
    tikzplotlib._cleanfigure._move_points_closer = _move_points_closer_fix
    tikzplotlib._cleanfigure._segment_visible = _segment_visible_fix
    tikzplotlib._cleanfigure._remove_NaNs = _remove_NaNs_fix
