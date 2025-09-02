from __future__ import annotations
from .methods import CalibMethod, LinearRegression
import numpy as np
from typing import Union, Optional

class CalibAPI():
  def __init__(self, method: Optional[CalibMethod] = None):
    """
    """
    match method:
      case CalibMethod() | None:
        self._method = method
      case _:
        raise TypeError(f"Expected method to be CalibMethod derived object, got {type(method).__name__}")

  @property
  def method(self) -> Optional[CalibMethod]:
    """
    """
    return self._method

  @method.setter
  def method(self, method: Optional[CalibMethod] = None) -> None:
    match method:
      case CalibMethod() | None:
        self._method = method
      case _:
        raise TypeError(f"Expected method to be CalibMethod derived object, got {type(method).__name__}")
  
  def calibrate(self, x, y):
    """
    @TODO: Why I use return res ??
    """
    match self._method:
      case CalibMethod():
        print("(CalibAPI::calibrate) called")
        self._method.fit(x, y)
        # res = self._method.fit(x, y)
        # return res
      case None:
        raise ValueError("Can't calibrate without a calibration method.")

  def predict(self, raw_val) -> Union[int, float, list, tuple, np.ndarray]:
    """ 
    calculate the true values using calibration method parameters from a raw value
    """
    match self._method:
      case CalibMethod():
        return self._method(raw_val)
      case None:
        raise ValueError("Can't predict without a calibration method.")

  def parameters(self):
    """
    Method to receive the calibration parameters.
    """
    match self._method:
      case CalibMethod():
        return self._method.params()
      case None:
        raise ValueError("No calibration method is set to get parameters from.")

  def export_params(self, filepath: str):
    match self._method:
      case CalibMethod():
        self._method.export_params(filepath)
      case None:
        raise ValueError("No calibration method is set to export parameters from.")


  def import_params(self, filepath: str):
    match self._method:
      case CalibMethod():
        self._method.import_params(filepath)
      case None:
        raise ValueError("No calibration method is set to import parameters into.")

  # def import_params(self, params_d):
  #   self._method.import_params(params_d)

  def __repr__(self):
    match self._method:
      case LinearRegression():
        # return f"CalibAPI::Linear{self.parameters()}"
        return f"{self._method}"
