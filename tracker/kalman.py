class Kalman1D:
    def __init__(self, initial: float, q: float, r: float):
        self.x = initial
        self.v = 0.0
        self.P00 = 1.0
        self.P01 = 0.0
        self.P10 = 0.0
        self.P11 = 1.0
        self.q = q
        self.r = r

    def predict(self):
        self.x += self.v
        P00 = self.P00 + self.P01 + self.P10 + self.P11 + self.q
        P01 = self.P01 + self.P11
        P10 = self.P10 + self.P11
        P11 = self.P11 + self.q
        self.P00, self.P01, self.P10, self.P11 = P00, P01, P10, P11

    def update(self, z: float):
        y = z - self.x
        S = self.P00 + self.r
        K0 = self.P00 / S
        K1 = self.P10 / S
        oldP00, oldP01, oldP11 = self.P00, self.P01, self.P11
        self.x += K0 * y
        self.v += K1 * y
        self.P00 = oldP00 - K0 * oldP00
        self.P01 = oldP01 - K0 * oldP01
        self.P10 = self.P10 - K1 * oldP00
        self.P11 = oldP11 - K1 * oldP01