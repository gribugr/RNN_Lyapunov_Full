from typing import NamedTuple
import math


class Quaternion(NamedTuple):
    w: float
    x: float
    y: float
    z: float

    def __add__(self, other):
        if isQuaternion(other):
            return Quaternion(*[k + v for k, v in zip(self, other)])
        else:
            raise TypeError("Quaternion should be given")
    __radd__ = __add__

    def __sub__(self, other):
        if isQuaternion(other):
            return Quaternion(*[k - v for k, v in zip(self, other)])
        else:
            raise TypeError("Quaternion should be given")

    def __rsub__(self, other):
        if isQuaternion(other):
            return Quaternion(*[k - v for k, v in zip(other, self)])
        else:
            raise TypeError("Quaternion should be given")

    def __mul__(self, other):
        if isNumber(other):
            return Quaternion(self.w * other, self.x * other, self.y * other, self.z * other)
        elif isQuaternion(other):
            w = (self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z)
            x = (self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y)
            y = (self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x)
            z = (self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w)
            return Quaternion(w=w, x=x, y=y, z=z)
        else:
            raise ValueError('Not yet implemented')

    def __rmul__(self, other):
        if isNumber(other):
            return Quaternion(self.w * other, self.x * other, self.y * other, self.z * other)
        return self.__class__(other) * self

    def __div__(self, other):
        if isNumber(other):
            return Quaternion(*[k / (1. * other) for k in self])
        else:
            raise TypeError("Number should be given")
    __truediv__ = __div__

    def __rdiv__(self, other):
        raise ValueError('Division by quaternions is not allowed')

    def __eq__(self, other):
        """Returns true if the following is true for each element:
        `absolute(a - b) <= (atol + rtol * absolute(b))`
        """
        if not isinstance(other, type(self)):
            return NotImplemented

        r_tol = 1.0e-13
        a_tol = 1.0e-14
        return all(math.fabs(item - iota) <= (a_tol + r_tol * math.fabs(iota)) for item, iota in zip(self, other))

    def __len__(self):
        return 4

    @property
    def scalar(self) -> float:
        return self.w

    @property
    def conjugate(self):
        return Quaternion(w=self.w, x=self.x * -1., y=self.y * -1., z=self.z * -1.)

    @property
    def norm(self):
        return math.sqrt(sum(map(lambda x: math.pow(x, 2), [k for k in self])))

    @property
    def normalized(self):
        n = self.norm
        if n == 0:
            raise ZeroDivisionError("Can't normalize with a zero quaternion")
        return self / n

    @property
    def inverse(self):
        return self.conjugate / math.pow(self.norm, 2)

    @staticmethod
    def empty():
        return Quaternion(w=1, x=0.0, y=0.0, z=0.0)

    @property
    def to_euler_angles(self):
        roll = math.atan2(2 * (self.w * self.x + self.y * self.z), 1 - 2 * (self.x * self.x + self.y * self.y))
        
        sinp = 2 * (self.w * self.y - self.z * self.x)
        if (abs(sinp) >= 1):
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        
        yaw = math.atan2(2 * (self.w * self.z + self.x * self.y), 1 - 2 * (self.y * self.y + self.z * self.z))
        
        return [roll, pitch, yaw]
    
    @staticmethod
    def from_str(input_str):
        temp = [float(i) for i in input_str.strip('()').split(',')]
        return Quaternion(x=temp[0], y=temp[1], z=temp[2], w=temp[3])
        

def isQuaternion(item) -> bool:
    return isinstance(item, Quaternion)


def isNumber(item) -> bool:
    return isinstance(item, (int, float))
