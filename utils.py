import numpy as np

def skew(vector):
    """This function returns a numpy array with the skew symmetric cross product matrix for vector.
    The skew symmetric cross product matrix is defined such that np.cross(a, b) = np.dot(skew(a), b)

    Args:
        vector (numpy.ndarray): An array like vector to create the skew symmetric cross product matrix for
    """
    return np.array([[0, -vector.item(2), vector.item(1)],
                    [vector.item(2), 0, -vector.item(0)],
                    [-vector.item(1), vector.item(0), 0]])