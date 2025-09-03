import unittest
import numpy as np
import json
from sensorcalibrationlib import CalibAPI, CalibMethod, LinearRegression

class CalibAPITest(unittest.TestCase):

  def setUp(self):
    self.x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0] 
    self.y = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0] 
    self.api = CalibAPI()
    self.api2 = CalibAPI(LinearRegression())
    # super().setUp()

  # def test_init(self):
  #   api = CalibAPI()

  def test_init(self):
    # check error initializations
    self.assertRaises(TypeError, lambda: CalibAPI(set()))
    self.assertRaises(TypeError, lambda: CalibAPI([1,2,3]))
    self.assertRaises(TypeError, lambda: CalibAPI("blabla"))
    with self.assertRaises(TypeError):
      self.api.method = dict() 

    # check correct init
    api = CalibAPI(LinearRegression())
    self.assertIsInstance(api.method, CalibMethod)
    self.assertIsInstance(api.method, LinearRegression)

  def test_calibrate(self):
    # check errors
    self.assertRaises(ValueError, lambda: self.api.calibrate(self.x, self.y))
    wrong_type_inp = set()
    wrong_len_inp = [1, 2, 3, 4]
    self.assertRaises(Exception, lambda: self.api2.calibrate(wrong_type_inp, self.y))
    self.assertRaises(Exception, lambda: self.api2.calibrate(wrong_len_inp, self.y))
    self.assertRaises(TypeError, lambda: self.api2.calibrate()) # no args

    self.api.method = LinearRegression()
    self.api.calibrate(self.x, self.y)
    np.testing.assert_allclose(self.api.parameters(), [2, 2], rtol=1e-12, atol=0)

    fibx = [1, 1, 3, 5]
    fiby = [8, 13, 21, 34]
    self.api2.calibrate(fibx, fiby)
    np.testing.assert_allclose(self.api2.parameters(), [5.818182, 4.454545], rtol=1e-7, atol=1e-7)

  def test_parameters(self):
    # check getting params with no method set
    self.assertRaises(ValueError, lambda: self.api.parameters())
    
    # check parameter import
    self.api2.import_params('param_files/api_test_0.json') # contains 'a': 15, 'b':3
    self.assertEqual(self.api2.parameters(), (15, 3))
    np.testing.assert_array_equal(self.api2.parameters(), self.api2.method.p.coef[::-1])

  def test_predict(self):
    # check error behavior
    self.assertRaises(ValueError, lambda: self.api.predict(3)) # no method
    self.assertRaises(ValueError, lambda: self.api2.predict(3)) # method, no params

    self.api2.calibrate(self.x, self.y)
    # wrong type arg
    self.assertRaises(TypeError, lambda: self.api2.predict({1,2,3}))
    self.assertRaises(TypeError, lambda: self.api2.predict(object))
    # empty arg
    self.assertEqual(self.api2.predict([]), [])

    # check prediction with all possible types
    self.assertEqual(self.api2.predict(10), 22)
    self.assertEqual(self.api2.predict(np.array([20, 30, 40])), [42, 62, 82])
    self.assertEqual(self.api2.predict((100, 200)), [202, 402])
    self.assertEqual(self.api2.predict([16, 17.]), [34, 36])

  def test_method_setup(self):
    """
    I might need a QuadRegression first for this..
    """
    pass

  def test_import_params(self):
    # check importing without method set
    self.assertRaises(ValueError, lambda: self.api.import_params('param_files/api_test_0.json'))

    self.api.method = LinearRegression()
    self.api.import_params('param_files/api_test_0.json')
    self.assertEqual(self.api.parameters(), (15, 3))

    # check importing from file with wrong number of params
    self.assertRaises(ValueError, lambda: self.api.import_params('param_files/api_test_import_one_key.json'))
    self.assertRaises(ValueError, lambda: self.api.import_params('param_files/api_test_import_three_key.json'))
    # check importing from file with wrong keys 
    self.assertRaises(ValueError, lambda: self.api.import_params('param_files/api_test_import_wrong_keys.json'))
    # check importing from non existent file 

    # check that params actually change after import
    self.api2.calibrate(self.x, self.y)
    np.testing.assert_allclose(self.api2.parameters(), (2, 2), rtol=1e-12, atol=0)
    self.api2.import_params('param_files/api_test_0.json')
    self.assertEqual(self.api2.parameters(), (15, 3)) # check they got the new values
    np.testing.assert_array_equal(self.api2.parameters(), self.api2.method.p.coef[::-1]) # check the new values broadcast to the Polynomial 
    np.testing.assert_array_equal((15, 3), self.api2.method.p.coef[::-1]) # sanity

  def test_export_params(self):
    # check error
    self.assertRaises(ValueError, self.api.export_params, "param_files/api_test_1.json") # no method set yet
    self.assertRaises(ValueError, self.api2.export_params, "param_files/api_test_1.json") # not calibrated yet

    output_file = "param_files/api_test_1.json"
    self.api2.calibrate(self.x, self.y)
    with open(output_file, encoding='utf-8') as f:
      contents = json.load(f)
    np.testing.assert_allclose(list(contents.values()), [2, 2], rtol=1e-12, atol=1e-12) # file should now contain 'a':2, 'b':2
