# Sensor Calibration Library

## Installation

You only have to install [uv](https://docs.astral.sh/uv/).

```bash
curl -LsSf https://astral.sh/uv/0.8.13/install.sh | sh
```

or

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

If you have any trouble visit [uv docs](https://docs.astral.sh/uv/getting-started/installation/).

## Running tests

```bash
uv run python -m unittest -v test/test*
```

## Implementation details

- `Python 3.13.3`, `uv 0.8.13`
- Backend `numpy 2.3.2`
- Everything was run/tested on Artix Linux 6.15.2 and Ubuntu 22.04

## TODO

- [ ] Use logging for warn messages and logs.
- [ ] Add dunder method operations for CalibMethod objects ??
- [ ] Add unittest.mock tests.
