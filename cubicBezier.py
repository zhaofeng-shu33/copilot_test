import numpy as np

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
