# Sensor Calibration Library

## Example Usage

```py
from sensorcalibrationlib import CalibAPI, LinearFit, QuadraticFit

x = [1,2,3]
y = (2,4,6)

api = CalibAPI(LinearFit())
api.calibrate(x, y)

api.method = QuadraticFit()
api.receive_calibration_parameters(3, 4, 5) # 3*x**2 + 4*x + 5

api.predict(20)
api.predict([100, 3, 1])
api.export_params("param_files/export_file.json") # this will be created

```

## Installation

You only have to install [uv](https://docs.astral.sh/uv/). Run

```bash
curl -LsSf https://astral.sh/uv/0.8.13/install.sh | sh
```

or

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

If you have any trouble with the installation visit [uv Installation docs](https://docs.astral.sh/uv/getting-started/installation/).

Run the following commands

```bash
git clone https://github.com/kpetrakis/SensorCalibrationLib
cd SensorCalibrationLib
uv sync
```

and you are good to go. Running

```bash
uv tree
```

you should see

```bash
sensorcalibrationlib v0.1.0
└── numpy v2.3.2
```


## Running example

Run

```bash
uv run main.py
```

Example output:

```bash
CalibAPI withoud model : CalibAPI(method = None)
CalibAPI with Linear model : CalibAPI(method = LinearFit(params = None))
CalibAPI with Linear model after calibration : CalibAPI(method = LinearFit(params = (2.0000000000000013, -2.051160198809135e-15)))
Calibration parameters: (2.0000000000000013, -2.051160198809135e-15)
================================
CalibAPI with new parameters : CalibAPI(method = LinearFit(params = (3.0, 4.0)))
Calibration parameters after received from user: (3.0, 4.0)
================================
CalibAPI.predict(4): 16.0
CalibAPI.predict([10, 20]): [34.0, 64.0]
================================
CalibAPI after changing model to Quadratic : CalibAPI(method = QuadraticFit(params = None))
CalibAPI with Quadratic model after calibration on (x+1)**2 data : CalibAPI(method = QuadraticFit(params = (2.000000000000001, 0.9999999999999996, 0.9999999999999996))) 
CalibAPI with Quadratice model parameters : (2.000000000000001, 0.9999999999999996, 0.9999999999999996)
================================
CalibAPI with Quadratice model after import : CalibAPI(method = QuadraticFit(params = (2.0, 3.0, 10.0)))
```

Feel free to modify `main.py` or add your own scripts to check the functionality.

## Running tests

```bash
uv run python -m unittest -v test/test*
```

## Requirements

- `Python 3.13`
- `numpy 2.3.2`

Everything was run/tested on Artix Linux 6.15.2 and Ubuntu 22.04

## TODO / POSSIBLE IMPROVEMENTS

- [ ] Add support for other polynomial familes (e.g. Chebysev, Legendre)
- [ ] Add the ability to define parameters at CalibMethod creation (e.g. `LinearFit(2,3)`)
- [ ] Check for numerical Stability.
- [ ] Use logging for warn messages and logs.
- [ ] Add dunder method operations (`__add__`, `__div__`, ..) for CalibMethod objects ??
- [ ] Add unittest.mock tests.
