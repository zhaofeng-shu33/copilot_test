import numpy as np
from geomdl import BSpline
from geomdl import operations
from geomdl.visualization import VisMPL
from scipy.integrate import quad
import matplotlib
curve = BSpline.Curve(normalize_kv=False)
   
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

def arc_len(curve, t0=0, t1=1, tol=0.01):
    """
    Compute the arc length of the curve from 0 to t.
    
    Parameters:
    - t: The parameter value up to which the arc length is computed (0 <= t <= 1).
    - tol: The tolerance for the numerical integration.
    
    Returns:
    - The arc length from 0 to t.
    """
    # Define the integrand as the norm of the derivative
    def integrand(u):
        return np.linalg.norm(curve.derivatives(u, order=1)[1])
    
    # Use scipy's quad function to integrate ||r'(t)|| from 0 to t
    arc_length, _ = quad(integrand, t0, t1, epsabs=tol)
    return arc_length
# Set degree
curve.degree = 3
curve.ctrlpts = [[5.0, 5.0,0], [10.0, 10.0, 0], [27.5, 20,0], 
                 [45.0, 10.0,0], [50.0, 5.0, 0]]
for i in curve.ctrlpts:
    i[0] *= 10
    i[1] *= 10
knotvector = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
#curve.sample_size = 120

end_ = 2
for i in range(len(knotvector)):
    knotvector[i] *= end_
curve.knotvector = knotvector
print(curve.evaluate_single(-1))
print(curve.evaluate_single(1))

a1 = curve.derivatives(0, order=1)[1]
a2 = curve.derivatives(0, order=2)[2]
a3 = curve.derivatives(0, order=3)[3]
print(a1,a2)

print(np.linalg.norm(curve.derivatives(-1, order=1)[1]))
print(np.linalg.norm(curve.derivatives(1, order=1)[1]))
print(operations.tangent(curve, -1)[1])
print(operations.tangent(curve, 1)[1])
print(1/curvature(curve,-1))
print(1/curvature(curve,1))
print(normal_unit_vector(curve,-1))
print(normal_unit_vector(curve,1))

print(curve.evaluate_single(0))
print(arc_len(curve, -1, 1))

vis = VisMPL.VisCurve3D()

fname = "test-curve.png"
# Evaluate curve
curve.evaluate()
curve.vis = vis
curve.render(filename=fname)


print(curve.evaluate_single(0.5))