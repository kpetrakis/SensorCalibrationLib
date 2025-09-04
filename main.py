from sensorcalibrationlib import CalibAPI, LinearFit, QuadraticFit

if __name__ == "__main__":
  
  x = [1,2,3]
  y = (2,4,6)

  api = CalibAPI()
  print(f"CalibAPI withoud model : {api}")

  api = CalibAPI(LinearFit())
  print(f"CalibAPI with Linear model : {api}")

  api.calibrate(x, y)
  print(f"CalibAPI with Linear model after calibration : {api}")
  print(f"Calibration parameters: {api.parameters()}")

  print(f"================================")
  api.receive_calibration_parameters(3, 4) # 3*x + 4
  print(f"CalibAPI with new parameters : {api}")
  print(f"Calibration parameters after received from user: {api.parameters()}")

  # predict
  print(f"================================")
  print(f"CalibAPI.predict(4): {api.predict(4)}")
  print(f"CalibAPI.predict([10, 20]): {api.predict([10, 20])}")

  # model modification
  print(f"================================")
  api.method = QuadraticFit()
  print(f"CalibAPI after changing model to Quadratic : {api}")
  # print(f"CalibAPI.predict(4): {api.predict(4)}")

  # 2*x**2 + x + 1
  x = [-2, -1, 0, 1, 2]
  y = [7, 2, 1, 4, 11]
  api.method = QuadraticFit()
  api.calibrate(x, y)
  print(f"CalibAPI with Quadratic model after calibration on (x+1)**2 data : {api}")
  print(f"CalibAPI with Quadratice model parameters : {api.parameters()}")

  print(f"================================")
  # import
  api.import_params("param_files/main.json")
  print(f"CalibAPI with Quadratice model after import : {api}")

  # export
  api.method = LinearFit()
  api.receive_calibration_parameters(5, 10)
  api.export_params("param_files/main_export.json") # this will be created


