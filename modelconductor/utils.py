__package__ = "modelconductor"
import numpy as np


class Measurement(dict):
    """
    A key-value mapping representing a dataframe measured at a single
    point in time from a data source
    """

    def to_numpy(self, keys=None):
        """

        Args:
            keys: Ordered list, a subset of the Measurement object's dict keys
            to be taken in to account when creating the numpy array

        Returns:
            numpy_meas: A subset of the measurement as a numpy array

        """
        if keys is None:
            keys = self.keys()
        numpy_meas = np.array([self[k] for k in keys], ndmin=2)
        return numpy_meas


class ModelResponse(dict):
    """
    A key-value mapping of dataframe returned from the model
    """