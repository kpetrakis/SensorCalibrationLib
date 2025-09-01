from sensorcalibrationlib import CalibAPI, LinearRegression
import numpy as np

# np.set_printoptions(precision=16, suppress=False)

if __name__ == "__main__":

  print("MAIN ENTERED")

  x = (1,2,3)
  y = (2,4,6)
  lr = LinearRegression()

  # api = CalibAPI(lr)
  api = CalibAPI(LinearRegression())

  # print(api.parameters())

  api.calibrate(x, y)
  print(api.method._p.coef)
  params = api.parameters()
  print("params", params)
  print(params[0] == 0)


  # new = np.array((5,))
  # p = np.polynomial.Polynomial.fit(x, y, deg=1, domain=[-1,1])
  # print("p", p)
  # print("p.domain", p.domain)
  # print("p.coef", p.coef)
  # print("p(new) = ", p(new))
