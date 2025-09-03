from __future__ import annotations
from .methods import CalibMethod, LinearRegression
from .types import CalibInputType, PredictInputType
import numpy as np
from typing import Union, Optional, List, Tuple

class CalibAPI():
  def __init__(self, method: Optional[CalibMethod] = None):
    """
    Initialize the CalibAPI instance.

    Parameters
    ----------
    method : Optional[CalibMethod]
      The calibration method to use. If None no operations can run (default is None).
    """
    match method:
      case CalibMethod() | None:
        self._method = method
      case _:
        raise TypeError(f"Expected method to be CalibMethod derived object, got {type(method).__name__}")

  @property
  def method(self) -> Optional[CalibMethod]:
    return self._method

  @method.setter
  def method(self, method: Optional[CalibMethod] = None) -> None:
    match method:
      case CalibMethod() | None:
        self._method = method
      case _:
        raise TypeError(f"Expected method to be CalibMethod derived object, got {type(method).__name__}")
  
  def calibrate(self, x: CalibInputType, y: CalibInputType) -> None:
    """
    Performs calibration using the instance's calibration method.

    Parameters
    ----------
    x : CalibInputType
      Input data for calibration.
    y : CalibInputType
      Target data for calibration.

    Raises
    ------
    ValueError
      If no calibration method is set (`self._method` is None).
    """
    match self._method:
      case CalibMethod():
        self._method.fit(x, y)
      case None:
        raise ValueError("Can't calibrate without a calibration method.")

  def predict(self, raw_val: PredictInputType) -> Union[float, List[float]]:
    """ 
    Calculates the target values from a raw input using the configured calibration method.

    Parameters
    ----------
    raw_val : PredictInputType
      The raw input value(s) to run regression on.

    Returns
    -------
    float or List[float]
      The predicted y value(s) corresponding to the input.

    Raises
    ------
    ValueError
      If no calibration method is set (`self._method` is None).
    """
    match self._method:
      case CalibMethod():
        return self._method(raw_val)
      case None:
        raise ValueError("Can't predict without a calibration method.")

  def parameters(self) -> Optional[Tuple]:
    """
    Retrieve the current calibration parameters from the configured method.

    Returns
    -------
    Optional[Tuple]
      The calibration parameters (coeficients of Polynomial) returned by the calibration method.

    Raises
    ------
    ValueError
      If no calibration method is set (`self._method` is None)
    """
    match self._method:
      case CalibMethod():
        return self._method.params()
      case None:
        raise ValueError("No calibration method is set to get parameters from.")

  def export_params(self, filepath: str):
    """
    Export the current calibration parameters to a file.

    Parameters
    ----------
    filepath : str
      The path to the file where the parameters will be saved.

    Raises
    ------
    ValueError
      If no calibration method is set (`self._method` is None).
    """
    match self._method:
      case CalibMethod():
        self._method.export_params(filepath)
      case None:
        raise ValueError("No calibration method is set to export parameters from.")


  def import_params(self, filepath: str):
    """
    Import calibration parameters from a file.

    Parameters
    ----------
    filepath : str
      The path to the file from which parameters will be loaded.

    Raises
    ------
    ValueError
      If no calibration method is set (`self._method` is None).
    """
    match self._method:
      case CalibMethod():
        self._method.import_params(filepath)
      case None:
        raise ValueError("No calibration method is set to import parameters into.")

  def __repr__(self):
    return f"CalibAPI(method = {self._method})"
