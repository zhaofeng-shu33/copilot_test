import numpy as np
from scipy.integrate import quad

class CubicBezier:
    def __init__(self, P0, P1, P2, P3):
        """
        Initialize the cubic Bézier curve with control points P0, P1, P2, P3.
        Points can be 2D or 3D coordinates.
        """
        self.P0 = np.array(P0, dtype=float)
        self.P1 = np.array(P1, dtype=float)
        self.P2 = np.array(P2, dtype=float)
        self.P3 = np.array(P3, dtype=float)
    
    def point(self, t):
        """Compute the point on the curve at parameter t."""
        return (1 - t)**3 * self.P0 + 3 * (1 - t)**2 * t * self.P1 + 3 * (1 - t) * t**2 * self.P2 + t**3 * self.P3
    
    def derivative(self, t):
        """Compute the first derivative of the curve at parameter t."""
        return 3 * (1 - t)**2 * (self.P1 - self.P0) + 6 * (1 - t) * t * (self.P2 - self.P1) + 3 * t**2 * (self.P3 - self.P2)
    
    def derivative_second(self, t):
        """Compute the second derivative of the curve at parameter t."""
        return 6 * (1 - t) * (self.P0 - 2*self.P1 + self.P2) + 6 * t * (self.P1 - 2*self.P2 + self.P3)
    
    def curvature(self, t):
        """Compute the curvature at parameter t."""
        d1 = self.derivative(t)
        d2 = self.derivative_second(t)
        cross = np.cross(d1, d2)
        numerator = np.linalg.norm(cross)
        denominator = np.linalg.norm(d1) ** 3
        return numerator / denominator if denominator != 0 else 0.0
    
    def normal_unit_vector(self, t):
        """Compute the unit normal vector at parameter t."""
        d1 = self.derivative(t)
        if len(d1) == 2:
            # 2D case: rotate tangent 90 degrees
            nx, ny = -d1[1], d1[0]
            magnitude = np.linalg.norm(d1)
            return np.array([nx, ny]) / magnitude if magnitude != 0 else np.array([0.0, 0.0])
        else:
            # 3D case: principal normal vector
            d2 = self.derivative_second(t)
            proj = (np.dot(d1, d2) / np.dot(d1, d1)) * d1 if np.dot(d1, d1) != 0 else 0
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

    def interpolate_2d(self, tx, Px, ty, Py):
        """
        Modify self.P1 and self.P2 such that self.point(tx) = Px and self.point(ty) = Py.
        """
        # Coefficients for the Bézier curve equation at tx
        B0_tx = (1 - tx)**3
        B1_tx = 3 * (1 - tx)**2 * tx
        B2_tx = 3 * (1 - tx) * tx**2
        B3_tx = tx**3

        # Coefficients for the Bézier curve equation at ty
        B0_ty = (1 - ty)**3
        B1_ty = 3 * (1 - ty)**2 * ty
        B2_ty = 3 * (1 - ty) * ty**2
        B3_ty = ty**3

        # Linear system to solve for P1 and P2
        A = np.array([
            [B1_tx, B2_tx],
            [B1_ty, B2_ty]
        ])
        B = np.array([
            Px - (B0_tx * self.P0 + B3_tx * self.P3),
            Py - (B0_ty * self.P0 + B3_ty * self.P3)
        ])

        # Solve the linear system
        P1_P2 = np.linalg.solve(A, B)

        # Update P1 and P2
        self.P1 = P1_P2[0]
        self.P2 = P1_P2[1]

    def arc_len(self, t, tol=0.01):
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
            return np.linalg.norm(self.derivative(u))
        
        # Use scipy's quad function to integrate ||r'(t)|| from 0 to t
        arc_length, _ = quad(integrand, 0, t, epsabs=tol)
        return arc_length

if __name__ == '__main__':
    a = 43.654448687742594
    b = 23.892977503339033
    alpha = 1.0471975511965976
    beta = 0.7853981633974483
    import numpy as np
    P0=np.array([0,0,0])
    P1=np.array([a*np.cos(alpha),a*np.sin(alpha),0])
    P2=np.array([100-b*np.cos(beta),b*np.sin(beta),0])
    P3=np.array([1000,0,0])
    cB = CubicBezier(P0,P1,P2,P3)
  
    # invoke the method of interpolation
    Py = [65.5389, 89.6769, 0]
    Px = [38.8343,58.1767, 0]
    cB.interpolate_2d(0.3, np.array(Px), 0.4, np.array(Py))
    print(cB.P1, cB.P2)
    a = cB.derivative(0)
    print(np.arctan(a[1]/a[0]) * 180 / np.pi)
    print(cB.point(0.0), cB.point(1.0))
    print(cB.point(0.3))
    print(cB.arc_len(0.3))
    print(cB.arc_len(0.4))