# Ellipse Fitting

A noisy data is sampled from a 2-D ellipse. The task is to estimate parameters of this ellipse from the data.

The correspondences on the parametric curve are simultaneously minimized along with the ellipse parameters (`theta` in code) using Leverberg-Marquardt nonlinear optimizer.

## Steps
1. Visualize data
```python
python data.py
```

2. Fit ellipse to data
```python
python minimize.py
```
