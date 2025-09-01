from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
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
    if self._p is not None:
      if isinstance(raw_val, (int, float, tuple, list, np.ndarray)):
        return self._p(raw_val)
      else:
        raise TypeError("__call__ type error")
    else:
      raise ValueError("__call__ value error")

  def params(self) -> tuple:
    """
    return polynomial coefficients in descending order
    e.g. for 2*x + 3 -> (2, 3)
    """
    if self._p is None:
      raise ValueError("Linear Calibrator not yet Calibrated. Run calibration, or import parameters.")
    else:
      # b, a = self._p.coef[0], self._p.coef[1] # convert ??
      # return map(lambda x: x.item(), (a, b))
      return tuple(map(lambda x: x.item(), self._p.coef[::-1])) # return

  def import_params(self, params_d: Dict[str, float]):
    pass

  def export_params(self) -> Dict[str, float]:
    pass


