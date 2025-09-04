from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
import json
from pathlib import Path
import warnings
from typing import Optional, Union, List, Tuple
from .types import CalibInputType, PredictInputType

class CalibMethod(ABC):
  """
  Abstract Base Class for calibration methods.

  This class defines the interface that all calibration methods must implement.
  """
  def __init__(self):
    '''
    @TODO: maybe receive coef as arguments ??
    '''
    pass

  @abstractmethod
  def fit(self, x: CalibInputType, y: CalibInputType) -> None:
    """
    Use np.polynomial.Polynomial as a backend to fit input data.

    Parameters
    ----------
    x : CalibInputType
      Input data for calibration.
    y : CalibInputType
      Target data for calibration.
    """
    pass

  @abstractmethod
  def __call__(self, raw_val: PredictInputType) -> Union[float, List[float]]:
    """
    Use np.polynomial.Polynomial backend to calculate the response of the Regression model.

    Parameters
    ----------
    raw_val : PredictInputType
      Raw input value(s) for which prediction is requested.

    Returns
    -------
    float or List[float]
      Regression result.
    """
    pass

  @abstractmethod
  def set_params(self, *arg) -> None:
    """
    Set polynomial parameters to the specified values.

    Parameters
    ----------
    *args : tuple or list, or separate numeric arguments
      The polynomial coefficients in descending order of degree.
      For example, for 3*x + 2, call set_params(3, 2) or set_params((3, 2)).

    Raises
    ------
    ValueError
      If the number of parameters is not exactly 2 or the input format is incorrect.
    """
    pass

  @abstractmethod
  def params(self):
    """
    Return polynomial coefficients in descending degree order.

    E.g. for the polynomial 2*x + 3, this method returns (2, 3).

    Returns
    -------
    Tuple[float, float]
      The coefficients of the polynomial, from highest to lowest degree.

    Raises
    ------
    ValueError
      If the calibrator has not yet been fitted or parameters imported.
    """
    pass

  @abstractmethod
  def import_params(self, filepath: str):
    """
    Import calibration parameters from a JSON file and set up the np.polynomial.Polynomial backend coeficients.

    Parameters
    ----------
    filepath : str
      The path to the JSON file containing calibration parameters. 
      The file must contain exactly as many keys as the degree of np.polynomial.Polynomial,
      with numerical (int or float) values, representing the coefficients of the polynomial.

    Raises
    ------
    FileNotFoundError
      If the specified file does not exist.
    ValueError
      If the file contents do not match the expected format.
    Exception
      Propagates any exceptions encountered during file reading or parsing.
    """
    pass

  @abstractmethod
  def export_params(self, filepath: str):
    """
    Export the current np.polynomial.Polynomial backend coefficients to a JSON file. 

    Parameters
    ----------
    filepath : str
      The destination file path where the calibration parameters will be saved.

    Raises
    ------
    ValueError
      If the calibration parameters have not been set (model not fitted or params not imported).
    
    """
    pass

  def __repr__(self):
    match self._p:
      case None:
        return f"{self.__class__.__name__}(params = None)"
      case np.polynomial.Polynomial():
        return f"{self.__class__.__name__}(params = {self.params()})"
      case _:
        # just for sanity
        raise TypeError(f"Polynomial p is of type {type(self._p).__name__}")  

class LinearFit(CalibMethod):
  """
  LinearFit models a linear calibration method using degree-1 polynomial fitting.

  e.g. y = a*x + b
  """
  def __init__(self):
    super().__init__()
    self._p : Optional[np.polynomial.Polynomial] = None

  @property
  def p(self) -> Optional[np.polynomial.Polynomial]:
    return self._p

  def fit(self, x: CalibInputType, y: CalibInputType) -> None:
    """
    Future Extension: We can use fit(.., full=True) to get diagnostic info and warn/act 
    according to rank(Vandermonde), singular values etc...
    """

    match (x, y):
      # TODO: expand that to raise a different error based on the specific condition not satisfied
      case (list() | tuple() | np.ndarray(), list() | tuple() | np.ndarray()) if len(x) > 2 and len(x) == len(y):
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        # self._p, diagnostic = np.polynomial.Polynomial.fit(x, y, deg=1, domain=[-1, 1], full=True)
        # resid, rank, sv, rcond = diagnostic
        self._p = np.polynomial.Polynomial.fit(x, y, deg=1, domain=[-1, 1])#.convert()
      case _:
        raise Exception("Give 2 equal len sequences of len > 2 and type list, tuple or np.ndarray.") # Type or Value ?

  def __call__(self, raw_val: PredictInputType) -> Union[float, list]:
    """
    given an empty list [], it will return []
    """
    if self._p is None:
      raise ValueError("LinearFit.__call__() called before Polynomial params are set. Fit or import them first.")

    if isinstance(raw_val, (int, float, tuple, list, np.ndarray)):
      np_res = self._p(raw_val) # returns np.array
      match np_res.size:
        case 1:
          return np_res.item()
        case _:
          return np_res.tolist()
    else:
      raise TypeError(f"LinearFit.__call__(x) expected x to be one of (int, float, tuple, list, np.array), got {type(raw_val).__name__}")

  def set_params(self, *args) -> None:
    """
    create polynomial on set or error ??
    """
    match len(args):
      case 1 if isinstance(args[0], (tuple, list)) and len(args[0]) == 2:
        self._p = np.polynomial.Polynomial(coef=args[0][::-1])
      case 2:
        a, b = args
        self._p = np.polynomial.Polynomial(coef=[b,a])
      case _:
        raise ValueError("For linear model, exactly 2 parameters must be provided (either in tuple, list or seperately).")
    

  def params(self) -> Tuple[float, float]:
    if self._p is None:
      raise ValueError("Linear Calibrator not yet Calibrated. Run calibration, or import parameters.")
    else:
      return tuple(map(lambda x: x.item(), self._p.coef[::-1]))

  def import_params(self, filepath: str) -> None:
    try:
      if (path := Path(filepath)).is_file():
        with open(path, "r") as f:
          params_d = json.load(f)

        # structural pattern matching to avoid sorting dict
        match params_d:
          case {"a": a, "b": b} if len(params_d) == 2 and isinstance(a, (int, float)) and isinstance(b, (int, float)):
            self._p = np.polynomial.Polynomial(coef=[b,a])
          case _:
            raise ValueError(f"{filepath} should only contain a, b keys with numerical values for a*x+b model.")
      else:
        raise FileNotFoundError(f"file {filepath} not found.")
    except Exception as e:
      raise

  def export_params(self, filepath: str) -> None:
    try:
      match self._p:
        case None:
          raise ValueError("Calibration params not set yet. Fit the model or import params first, before exporting")
        case np.polynomial.Polynomial():
          path = Path(filepath)
          if path.exists():
            # overwrite or Error ??
            warnings.warn(f"path {filepath} already exists, will be overwritten by export.")

          params_d = dict(zip(('a', 'b'), self.params()))
          with open(path, "w") as f:
            json.dump(params_d, f, indent=2)
    except Exception as e:
      raise
  

class QuadraticFit(CalibMethod):
  """
  QuadraticFit models a quadratic calibration method using degree-2 polynomial fitting.

  e.g. y = a*(x**2) + b*x + c
  """
  def __init__(self):
    super().__init__()
    self._p : Optional[np.polynomial.Polynomial] = None

  @property
  def p(self) -> Optional[np.polynomial.Polynomial]:
    return self._p

  def fit(self, x: CalibInputType, y: CalibInputType) -> None:
    match (x, y):
      case (list() | tuple() | np.ndarray(), list() | tuple() | np.ndarray()) if len(x) > 2 and len(x) == len(y):
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        self._p = np.polynomial.Polynomial.fit(x, y, deg=2, domain=[-1, 1])#.convert()
      case _:
        raise Exception("Give 2 equal len sequences of len > 2 and type list, tuple or np.ndarray.") # Type or Value ?

  def __call__(self, raw_val: PredictInputType) -> Union[float, list]:
    if self._p is None:
      raise ValueError("QuadraticFit.__call__() called before Polynomial params are set. Fit or import them first.")

    if isinstance(raw_val, (int, float, tuple, list, np.ndarray)):
      np_res = self._p(raw_val) # returns np.array
      match np_res.size:
        case 1:
          return np_res.item()
        case _:
          return np_res.tolist()
    else:
      raise TypeError(f"QuadraticFit.__call__(x) expected x to be one of (int, float, tuple, list, np.array), got {type(raw_val).__name__}")
  
  def params(self) -> Tuple[float, float, float]:
    if self._p is None:
      raise ValueError("Quadratic Calibrator not yet Calibrated. Run calibration, or import parameters.")
    else:
      return tuple(map(lambda x: x.item(), self._p.coef[::-1]))

  def set_params(self, *args) -> None:
    """
    create polynomial on set or error ??
    """
    match len(args):
      case 1 if isinstance(args[0], (tuple, list)) and len(args[0]) == 3:
        self._p = np.polynomial.Polynomial(coef=args[0][::-1])
      case 3:
        a, b, c = args
        self._p = np.polynomial.Polynomial(coef=[c,b,a])
      case _:
        raise ValueError("For quadratic model, exactly 3 parameters must be provided (either in tuple, list or seperately).")
  
  def import_params(self, filepath: str) -> None:
    try:
      if (path := Path(filepath)).is_file():
        with open(path, "r") as f:
          params_d = json.load(f)

        # structural pattern matching to avoid sorting dict
        match params_d:
          case {"a": a, "b": b, "c":c} if len(params_d) == 3 and isinstance(a, (int, float)) and isinstance(b, (int, float)) and isinstance(c, (int, float)):
            self._p = np.polynomial.Polynomial(coef=[c,b,a])
          case _:
            raise ValueError(f"{filepath} should only contain a, b, c keys with numerical values for a*x**2+b*x+c quadratic model.")
      else:
        raise FileNotFoundError(f"file {filepath} not found.")
    except Exception as e:
      raise
  
  def export_params(self, filepath: str) -> None:
    try:
      match self._p:
        case None:
          raise ValueError("Calibration params not set yet. Fit the model or import params first, before exporting")
        case np.polynomial.Polynomial():
          path = Path(filepath)
          if path.exists():
            # overwrite or Error ??
            warnings.warn(f"path {filepath} already exists, will be overwritten by export.")

          params_d = dict(zip(('a', 'b', 'c'), self.params()))
          with open(path, "w") as f:
            json.dump(params_d, f, indent=2)
    except Exception as e:
      raise