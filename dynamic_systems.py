

class Dynamics:
    @staticmethod
    def rossler(xyz, t0, a=0.5, b=2.0, c=4.0):
        x, y, z = xyz
        xdot = - y - z
        ydot = x + a*y
        zdot = b + z*(x - c)
        return xdot, ydot, zdot

    @staticmethod
    def lorenz(xyz, t0, a=10.0, b=28, c=8/3):
        x, y, z = xyz
        xdot = -a*x + a*y
        ydot = b*x - y - x*z
        zdot = x*y - c*z
        return xdot, ydot, zdot

    @staticmethod
    def modified_lorenz(xyz, t0, a=10.0, b=28, c=8/3):
        x, y, z = xyz
        xdot = -a*x + a*y
        ydot = b*x - y - x*z
        zdot = x*y - c*z + x
        return xdot, ydot, zdot
