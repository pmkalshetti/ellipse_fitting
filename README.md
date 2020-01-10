# Ellipse Fitting
A noisy data is sampled from a 2-D ellipse. The task is to estimate parameters of this ellipse from the data.

The correspondences on the parametric curve are simultaneously minimized along with the ellipse parameters using Leverberg-Marquardt (LM) nonlinear optimizer.
The LM implementation is inspired from [tensorflow_graphics](https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/math/optimizer/levenberg_marquardt.py).

Thanks to Andrew Fitzgibbon for this trick(http://www.fitzgibbon.ie/bmvc15_tutorial).

## Result
**True Parameters**: [3, 2, 5, 4, 5, 7]

**Fitted (using squared error) Parameters**: [1.96, 3.02, 5.01, 1.52, 5.21, 7.88]

**Fitted (using absolute error) Parameters**: [1.87, 2.66, 5.35, 1.59, 5.04, 8.06]

### Squared Error  Metric
![Squared Fitting Iterations](media_readme/squared.gif)

### Absolute Error  Metric
![Absolute Fitting Iterations](media_readme/absolute.gif)


## Setup
The code uses `python3`.
Assuming `python3-pip` is installed (specifically use a [virtual environment](https://docs.python.org/3/library/venv.html)), the dependencies can be installed using
```bash
pip install -r requirements.txt
```

## Running Code
#### Visualize Data
```python
python src/data.py
```
#### Fit ellipse to data
```python
python src/fit.py
```
