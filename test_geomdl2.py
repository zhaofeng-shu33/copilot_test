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
        d2 = np.array(curve.derivatives(t, order=2)[2])
        proj = float(np.dot(d1, d2) / np.dot(d1, d1)) * d1 if np.dot(d1, d1) != 0 else 0
        normal_vector = d2 - proj
        magnitude = np.linalg.norm(normal_vector)
        print(magnitude)
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

def arc_len(curve, t, tol=0.01):
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
    arc_length, _ = quad(integrand, 0, t, epsabs=tol)
    return arc_length
def get_right_hand(Q0, Q1, tangent, tension, principal_normal_v, curvature_radius):
    tangent_norm = np.linalg.norm(Q1 - Q0) * tension
    Qt0_d1 = tangent_norm * tangent
    Qt0_d2 = principal_normal_v / curvature_radius * tangent_norm ** 2
    Qt0 = Q0
    A = np.zeros([6,6])
    d = np.zeros(6)
    A[0, 0] = -0.5
    A[0, 2] = 0.5
    A[3, 3] = -0.5
    A[3, 5] = 0.5
    d[0] = Qt0_d1[0]
    d[3] = Qt0_d1[1]

    A[1, 0] = 1/6
    A[1, 1] = 2/3
    A[1, 2] = 1/6
    A[4, 3] = 1/6
    A[4, 4] = 2/3
    A[4, 5] = 1/6
    d[1] = Qt0[0]
    d[4] = Qt0[1]

    dx = tangent[0]
    dy = tangent[1]
    A[2, 0] = 1 - dx*dx
    A[2, 1] = -2 + 2 *dx*dx
    A[2, 2] = 1 - dx*dx
    A[2, 3] = - dx * dy
    A[2, 4] = 2 * dx * dy
    A[2, 5] = - dx * dy
    A[5, 0] = - dx * dy
    A[5, 1] = 2 * dx * dy
    A[5, 2] = - dx * dy
    A[5, 3] = 1 - dy*dy
    A[5, 4] = -2 + 2 * dy*dy
    A[5, 5] = 1 - dy*dy
    d[2] = Qt0_d2[0]
    d[5] = Qt0_d2[1]
    # Solve the linear system
    x = np.linalg.solve(A, d)
    P0_P1_P2 = np.array([[x[0], x[3]], [x[1], x[4]], [x[2], x[5]]])
    return P0_P1_P2

    """
    Compute the right-hand side of the equation.
    
    Parameters:
    - tension: The tension vector.
    - tangent: The tangent vector.
    - curvature: The curvature value.
    
    Returns:
    - The right-hand side of the equation.
    """
    return np.cross(tension, tangent) * curvature
# Set degree
curve.degree = 3
curve.ctrlpts = [[5.0, 5.0,0], [10.0, 10.0, 0], [20.0, 15.0, 0], 
                 [35.0, 15.0,0], [45.0, 10.0,0], [50.0, 5.0, 0]]
for i in curve.ctrlpts:
    i[0] *= 1
    i[1] *= 1
knotvector = [-1, -2/3, -1/3, 0.0, 1/3, 2/3, 1.0, 1.0 + 1/3, 1.0 + 2/3, 2.0]
#curve.sample_size = 120

end_ = 3
for i in range(len(knotvector)):
    knotvector[i] *= end_
curve.knotvector = knotvector
print(np.array(curve.evaluate_single(1)))
Q0 = np.array(curve.evaluate_single(0))
Q1 = np.array(curve.evaluate_single(end_))
tangent0 = np.array(operations.tangent(curve, 0)[1])
tension = np.linalg.norm(curve.derivatives(0, order=1)[1])/np.linalg.norm(Q1 - Q0)
curvature_radius_0 = 1/curvature(curve,0)
normal_vector_0 = normal_unit_vector(curve,0)
P0_P1_P2 = get_right_hand(Q0, Q1, tangent0, tension, normal_vector_0, curvature_radius_0)
print(P0_P1_P2[0, :])
print(P0_P1_P2[1, :])
print(P0_P1_P2[2, :])

