import numpy as np


class ManiuplatorModel:
    def __init__(self, Tp):
        self.Tp = Tp
        self.l1 = 0.5
        self.r1 = 0.04
        self.m1 = 3.
        self.l2 = 0.4
        self.r2 = 0.04
        self.m2 = 2.4
        self.I_1 = 1 / 12 * self.m1 * (3 * self.r1 ** 2 + self.l1 ** 2)
        self.I_2 = 1 / 12 * self.m2 * (3 * self.r2 ** 2 + self.l2 ** 2)
        self.m3 = 0.0    #0.1
        self.r3 = 0.05    #0.05
        self.I_3 = 2. / 5 * self.m3 * self.r3 ** 2

        self.d1 = self.l1 / 2
        self.d2 = self.l2 / 2



        self.alpha = self.m1 * self.d1 ** 2 + self.I_1 + self.m2 * (self.l1 ** 2 + self.d2 ** 2) + self.I_2 + self.m3 * (self.l1 ** 2 + self.l2 ** 2) + self.I_3
        self.beta = self.m2 * self.l1 * self.d2 + self.m3 * self.l1 * self.l2
        self.gamma = self.m2 * self.d2 ** 2 + self.I_2 + self.m3 * self.l2 ** 2 + self.I_3

    def M(self, x):
        q1, q2, q1_dot, q2_dot = x

        M = np.array([
                [self.alpha + 2 * self.beta * np.cos(q2), self.gamma + self.beta * np.cos(q2)],
                [self.gamma + self.beta * np.cos(q2), self.gamma]
        ])
        return M

    def C(self, x):
        q1, q2, q1_dot, q2_dot = x

        C = np.array([
            [-self.beta * np.sin(q2) * q2_dot, -self.beta * np.sin(q2) * (q1_dot + q2_dot)],
            [self.beta * np.sin(q2)* q1_dot, 0]
        ])
        return C
