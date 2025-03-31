import numpy as np
from geomdl import BSpline
from geomdl import operations
from geomdl.visualization import VisMPL
import matplotlib
matplotlib.use('agg')
curve = BSpline.Curve()
   
def normal_unit_vector(curve, t):
    """Compute the unit normal vector at parameter t."""
    d1 = np.array(curve.derivatives(t,order=1)[1])
    if len(d1) == 2:
        # 2D case: rotate tangent 90 degrees
        nx, ny = -d1[1], d1[0]
        magnitude = np.linalg.norm(d1)
        return np.array([nx, ny]) / magnitude if magnitude != 0 else np.array([0.0, 0.0])
    else:
        # 3D case: principal normal vector
        d2 = curve.derivatives(t, order=2)[2]
        proj = float(np.dot(d1, d2) / np.dot(d1, d1)) * d1 if np.dot(d1, d1) != 0 else 0
        normal_vector = d2 - proj
        magnitude = np.linalg.norm(normal_vector)
        if magnitude == 0:
            # Fallback to arbitrary perpendicular vector if necessary
            v = np.array([1, 0, 0])
            if np.allclose(np.cross(d1, v), 0):
                v = np.array([0, 1, 0])
            normal_vector = np.cross(d1, v)
            magnitude = np.linalg.norm(normal_vector)
        return normal_vector / magnitude if magnitude != 0 else np.zeros(3)

def curvature(curve, t):
    """Compute the curvature at parameter t."""
    d1 = np.array(curve.derivatives(t,order=1)[1])
    d2 =  np.array(curve.derivatives(t, order=2)[2])
    cross = np.cross(d1, d2)
    numerator = np.linalg.norm(cross)
    denominator = np.linalg.norm(d1) ** 3
    return numerator / denominator if denominator != 0 else 0.0
# Set degree
curve.degree = 3
curve.sample_size = 100
curve.ctrlpts = [[5.0, 5.0,0], [10.0, 10.0, 0], [20.0, 15.0, 0], 
                 [35.0, 15.0,0], [45.0, 10.0,0], [50.0, 5.0, 0]]

curve.knotvector = [-1, -2/3, -1/3, 0.0, 1/3, 2/3, 1.0, 1.0 + 1/3, 1.0 + 2/3, 2.0]

print(curve.evaluate_single(0))
print(curve.evaluate_single(1))
print(np.linalg.norm(curve.derivatives(0, order=1)[1]))
print(np.linalg.norm(curve.derivatives(1, order=1)[1]))
print(operations.tangent(curve, 0)[1])
print(operations.tangent(curve, 1)[1])
print(1/curvature(curve,0))
print(1/curvature(curve,1))
print(normal_unit_vector(curve,0))
print(normal_unit_vector(curve,1))
print(curve.evaluate_single(0.5))



vis = VisMPL.VisCurve3D()

fname = "test-curve.png"

curve.vis = vis
curve.render(filename=fname, plot=True)