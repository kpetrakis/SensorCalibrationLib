from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
import json
from pathlib import Path
import warnings
from typing import Dict, Optional, Union

class CalibMethod(ABC):
  def __init__(self):
    '''
    '''

  @abstractmethod
  def fit(self, x, y):
    pass

  @abstractmethod
  def __call__(self, raw_val: Union[int, float, list, tuple, np.ndarray]):
    pass

  @abstractmethod
  def params(self):
    pass

  @abstractmethod
  def import_params(self, filepath: str):
    pass

  @abstractmethod
  def export_params(self, filepath: str):
    pass

  # @abstractmethod
  # def import_params(self, params_d: Dict[str, float]):
  #   pass

  # @abstractmethod
  # def export_params(self) -> Dict[str, float]:
  #   pass

class LinearRegression(CalibMethod):
  """
  TODO: make _p a property ??
  """
  def __init__(self):
    super().__init__()
    self._p : Optional[np.polynomial.Polynomial] = None

  # @property
  # def p(self) -> Optional[np.polynomial.Polynomial]:
  #   return self._p

  def fit(self, x, y):
    """
    @TODO add checks for args that dims match, that len(x) > 2, etc..

    Future Extension: We can use fit(.., full=True) to get diagnostic info and warn/act 
    according to rank(Vandermonde), singular values etc...
    """
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    # self._p, diagnostic = np.polynomial.Polynomial.fit(x, y, deg=1, domain=[-1, 1], full=True)
    self._p = np.polynomial.Polynomial.fit(x, y, deg=1, domain=[-1, 1])#.convert()
    print("window", self._p.window)
    print("domain", self._p.domain)

  def __call__(self, raw_val: Union[int, float, list, tuple, np.ndarray]):
    if self._p is None:
      raise ValueError("__call__ value error")

    if isinstance(raw_val, (int, float, tuple, list, np.ndarray)):
      return self._p(raw_val)
    else:
      raise TypeError("__call__ type error")

  def params(self) -> tuple:
    """
    return polynomial coefficients in descending order
    e.g. for 2*x + 3 -> (2, 3)
    """
    if self._p is None:
      raise ValueError("Linear Calibrator not yet Calibrated. Run calibration, or import parameters.")
    else:
      return tuple(map(lambda x: x.item(), self._p.coef[::-1]))

  def import_params(self, filepath: str):
    """
    """
    try:
      if (path := Path(filepath)).is_file():
        with open(path, "r") as f:
          params_d = json.load(f)

        # structural pattern matching to avoid sorting dict
        match params_d:
          case {"a": a, "b": b} if len(params_d) == 2 and isinstance(a, (int, float)) and isinstance(b, (int, float)):
            # print(a, b)
            # print("json contains:", params_d)
            self._p = np.polynomial.Polynomial(coef=[b,a])
            print(self._p)
          case _:
            # print("json contains:", params_d)
            raise ValueError(f"{filepath} should only contain a, b keys with numerical values for a*x+b regression.")
      else:
        raise FileNotFoundError(f"file {filepath} not found.")
    except Exception as e:
      raise

  def export_params(self, filepath: str):
    try:
      if self._p is None:
        raise ValueError("Calibration params not set yet. Fit the model or import params first, before exporting")

      path = Path(filepath)
      if path.exists():
        # overwrite or Error ??
        warnings.warn(f"path {filepath} already exists, will be overwritten by export.")

      params_d = dict(zip(('a', 'b'), self.params()))
      with open(path, "w") as f:
        json.dump(params_d, f, indent=2)
    except Exception as e:
      raise

