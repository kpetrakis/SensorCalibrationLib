import numpy as np
import numpy.typing as npt
from typing import Union, List, Tuple

# the basic type operations like calibration, prediction operate on..
CalibInputType = Union[List[Union[int, float]], Tuple[Union[int, float], ...], npt.NDArray[Union[np.float32, np.int32, np.float64]]]
PredictInputType =  Union[int, float, List[Union[int, float]], Tuple[Union[int, float], ...], npt.NDArray[Union[np.float32, np.int32, np.float64]]]

# PredictOutType = Union[int, float, list, tuple, npt.NDArray[Union[np.float32, np.int32, np.float64]]]