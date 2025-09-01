from __future__ import annotations
from .methods import CalibMethod
import numpy as np
from typing import Dict, Union

class CalibAPI():
  def __init__(self, method:CalibMethod):
    """
    """
    if isinstance(method, CalibMethod):
      self._method = method
    else:
      raise TypeError(f"Expected method to be CalibMethod object, got {type(method).__name__}")

  @property
  def method(self) -> CalibMethod:
    """
    """
    return self._method

  @method.setter
  def method(self, method) -> None:
    self._method = method
  
  def calibrate(self, x, y):
    """
    """
    print("(CalibAPI::calibrate) called")
    res = self._method.fit(x, y)
    return res

  def predict(self, raw_val) -> Union[int, float, list, tuple, np.ndarray]:
    """ 
    calculate the true values using calibration method parameters from a raw value
    """
    return self._method(raw_val)

  def parameters(self):
    """
    Method to receive the calibration parameters.
    """
    return self._method.params()

  def export_params(self, filepath: str):
    pass

  def import_params(self, filepath: str):
    pass

  # def import_params(self, params_d):
  #   self._method.import_params(params_d)