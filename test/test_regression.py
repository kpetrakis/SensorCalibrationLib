import unittest
import numpy as np
from sensorcalibrationlib import LinearFit

class LinearFitTest(unittest.TestCase):

    def setUp(self):
      self.raw1 = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
      self.true1 = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0] 
      self.target_coef1 = (2, 2)

      self.raw_val_scalar_1 = 10.0
      self.raw_val_arr_1 = (6.0, 7.0)
      self.target_scalar_1 = 22.0
      self.target_arr_1 = [14.0, 16.0] 

      self.raw2 = [0, 10, 20, 30, 40]
      self.true2 = [100, 80, 60, 40, 20]
      self.target_coef2 = (-2, 100)

      self.raw_val_scalar_2 = 100.0
      self.raw_val_arr_2 = (50.0, 70.0)
      self.target_scalar_2 = -100.0
      self.target_arr_2 = [0.0, -40.0] 

      self.raw3 = [1.0, 10.0]
      self.true3 = [2.0, 20]

      self.lr = LinearFit()

    def test_init(self):
      self.assertEqual(self.lr.p, None)

    def test_fit(self):
      self.lr.fit(self.raw1, self.true1)
      # self.assertAlmostEqual(self.lr.params()[0], self.target_coef1[0], delta=1e-12)
      # self.assertAlmostEqual(self.lr.params()[1], self.target_coef1[1], delta=1e-12)
      np.testing.assert_allclose(self.lr.params(), self.target_coef1, rtol=1e-12, atol=0)
      # verify order of coeef
      np.testing.assert_allclose(self.lr.p.coef, self.target_coef1[::-1], rtol=1e-12, atol=0)

      self.lr.fit(self.raw2, self.true2)
      # self.assertAlmostEqual(self.lr.params()[0], self.target_coef2[0], delta=1e-12)
      # self.assertAlmostEqual(self.lr.params()[1], self.target_coef2[1], delta=1e-12)
      np.testing.assert_allclose(self.lr.params(), self.target_coef2, rtol=1e-12, atol=0)
      # verify order of coef
      np.testing.assert_allclose(self.lr.p.coef, self.target_coef2[::-1], rtol=1e-12, atol=0)

      self.assertRaises(Exception, lambda: self.lr.fit(self.raw3, self.true3))

      # check numerical stability
      # x = [1.79e307, 0.0001, 10]
      # y = [1.79e308, 4, 20] 
      # self.lr.fit(x, y)
      # print(self.lr.params())
      # # np.testing.assert_allclose()

    def test_predict(self):
      '''
      testing __call__ of LinearFit
      '''
      self.lr.fit(self.raw1, self.true1)
      self.assertEqual(self.lr(self.raw_val_scalar_1), self.target_scalar_1)
      np.testing.assert_allclose(self.lr(self.raw_val_arr_1), self.target_arr_1, rtol=1e-12, atol=1e-12)

      self.lr.fit(self.raw2, self.true2)
      self.assertAlmostEqual(self.lr(self.raw_val_scalar_2), self.target_scalar_2, delta=1e-12)
      np.testing.assert_allclose(self.lr(self.raw_val_arr_2), self.target_arr_2, rtol=1e-12, atol=1e-12)

      # check attempt prediction before fitting
      linear_reg = LinearFit()
      self.assertRaises(ValueError, linear_reg, 3)

      # check attempt prediction with invalid arg, e.g. lr("bla")
      self.assertRaises(TypeError, lambda: self.lr("blabla"))

    def test_params(self):
      # check calling params() on unfitted LinerRegression object
      self.assertRaises(ValueError, lambda: self.lr.params())

      x = [1, 2, 3]
      y = np.array([2, 4, 6]) # a=2, b=0
      self.lr.fit(x, y)
      np.testing.assert_allclose(self.lr.params(), [2.0, 0.0], rtol=1e-12, atol=1e-12)
      np.testing.assert_allclose(self.lr.p.coef[::-1], [2.0, 0.0], rtol=1e-12, atol=1e-12)

    def test_set_params(self):
      # check errors
      self.assertRaises(ValueError, lambda: self.lr.set_params(5, 4, 3))
      self.assertRaises(ValueError, lambda: self.lr.set_params(5))
      self.assertRaises(ValueError, lambda: self.lr.set_params([5]))
      self.assertRaises(ValueError, lambda: self.lr.set_params((5, 5, 5)))

      self.lr.set_params(5, 4)
      self.assertEqual(self.lr.params(), (5.0, 4.0))
      self.lr.set_params([15, 28])
      self.assertEqual(self.lr.params(), (15.0, 28.0))


    def test_import_params(self):
      # check import works as expected
      self.lr.import_params("param_files/linear_test_import.json")
      np.testing.assert_array_equal(self.lr.params(), [50, 10])

      # check importing from file with only one key (e.g. a)
      self.assertRaises(ValueError, lambda: self.lr.import_params("param_files/linear_test_import_one_key.json"))
      # check importing from file with three keys (e.g. a, b, c)
      self.assertRaises(ValueError, lambda: self.lr.import_params("param_files/linear_test_import_three_key.json"))
      # check importing from a non-existent file 
      self.assertRaises(FileNotFoundError, lambda: self.lr.import_params("param_files/blabla.json"))
    
    def test_export_params(self):
      # check export raises error when called before params are set either through fit or import
      self.assertRaises(ValueError, lambda: self.lr.export_params("param_files/linear_test_import_0.json"))
