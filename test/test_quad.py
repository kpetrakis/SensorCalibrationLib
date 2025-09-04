import unittest
import numpy as np
from sensorcalibrationlib import QuadraticFit

class QuadraticFitTest(unittest.TestCase):

  def setUp(self):
    self.raw1 = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    self.true1 = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0] 
    self.target_coef1 = (0, 2, 2)

    self.raw_val_scalar_1 = 10.0
    self.raw_val_arr_1 = (6.0, 7.0)
    self.target_scalar_1 = 22.0
    self.target_arr_1 = [14.0, 16.0] 

    # 2x**2 + x + 1 -> target coef 2, 1, 1
    self.x = [-2, -1, 0, 1, 2]
    self.y = [7, 2, 1, 4, 11]
    self.target_coef2 = (2, 1, 1)

    self.raw_val_scalar_2 = 100.0
    self.raw_val_arr_2 = (50.0, 70.0)
    self.target_scalar_2 = 20101.0
    self.target_arr_2 = [5051.0, 9871.0] 

    self.quad = QuadraticFit()

  def test_init(self):
    self.assertEqual(self.quad.p, None)

  def test_fit(self):
    self.quad.fit(self.raw1, self.true1)
    np.testing.assert_allclose(self.quad.params(), self.target_coef1, rtol=1e-12, atol=1e-12)
    # verify order of coef
    np.testing.assert_allclose(self.quad.p.coef, self.target_coef1[::-1], rtol=1e-12, atol=1e-12)

    self.quad.fit(self.x, self.y)
    np.testing.assert_allclose(self.quad.params(), self.target_coef2, rtol=1e-12, atol=1e-12)
    # verify order of coef
    np.testing.assert_allclose(self.quad.p.coef, self.target_coef2[::-1], rtol=1e-12, atol=1e-12)

    raw3 = [1,2]
    true3 = [1,2]
    self.assertRaises(Exception, lambda: self.quad.fit(raw3, true3))

  def test_predict(self):
    '''
    testing __call__ of QuadraticFit
    '''
    self.quad.fit(self.raw1, self.true1)
    self.assertAlmostEqual(self.quad(self.raw_val_scalar_1), self.target_scalar_1, delta=1e-10)
    np.testing.assert_allclose(self.quad(self.raw_val_arr_1), self.target_arr_1, rtol=1e-12, atol=1e-12)

    self.quad.fit(self.x, self.y)
    self.assertAlmostEqual(self.quad(self.raw_val_scalar_2), self.target_scalar_2, delta=1e-10)
    np.testing.assert_allclose(self.quad(self.raw_val_arr_2), self.target_arr_2, rtol=1e-12, atol=1e-12)

    # check attempt prediction before fitting
    quad_reg = QuadraticFit()
    self.assertRaises(ValueError, quad_reg, 3)

    # check attempt prediction with invalid arg, e.g. self.quad("bla")
    self.assertRaises(TypeError, lambda: self.quad("blabla"))

  def test_params(self):
    # check calling params() on unfitted QuadraticFit object
    self.assertRaises(ValueError, lambda: self.quad.params())

    x = [1, 2, 3]
    y = np.array([2, 4, 6]) # a=2, b=0
    self.quad.fit(x, y)
    np.testing.assert_allclose(self.quad.params(), [0.0, 2.0, 0.0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(self.quad.p.coef[::-1], [0.0, 2.0, 0.0], rtol=1e-12, atol=1e-12)
    self.quad.fit(self.x, self.y)
    np.testing.assert_allclose(self.quad.params(), [2.0, 1.0, 1.0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(self.quad.p.coef[::-1], [2.0, 1.0, 1.0], rtol=1e-12, atol=1e-12)

  def test_set_params(self):
    # check errors
    self.assertRaises(ValueError, lambda: self.quad.set_params(5, 4, 3, 4))
    self.assertRaises(ValueError, lambda: self.quad.set_params(5))
    self.assertRaises(ValueError, lambda: self.quad.set_params([5]))
    self.assertRaises(ValueError, lambda: self.quad.set_params((5, 5, 5, 5)))

    self.quad.set_params(5, 4, 3)
    self.assertEqual(self.quad.params(), (5.0, 4.0, 3.0))
    self.quad.set_params([15, 28, 32])
    self.assertEqual(self.quad.params(), (15.0, 28.0, 32.0))

  def test_import_params(self):
    # check import works as expected
    self.quad.import_params("param_files/quad_test_import.json")
    np.testing.assert_array_equal(self.quad.params(), [50, 10, 1])

    # check importing from file with only one key (e.g. a)
    self.assertRaises(ValueError, lambda: self.quad.import_params("param_files/quad_test_import_one_key.json"))
    # check importing from file with four keys (e.g. a, b, c, d)
    self.assertRaises(ValueError, lambda: self.quad.import_params("param_files/quad_test_import_four_key.json"))
    # check importing from a non-existent file 
    self.assertRaises(FileNotFoundError, lambda: self.quad.import_params("param_files/blabla.json"))

  def test_export_params(self):
    # check export raises error when called before params are set either through fit or import
    self.assertRaises(ValueError, lambda: self.quad.export_params("param_files/quad_test_import_0.json"))