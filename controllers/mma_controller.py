import numpy as np
from .controller import Controller
from models.manipulator_model import ManiuplatorModel

class MMAController(Controller):
    def __init__(self, Tp):
        # TODO: Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # I:   m3=0.1,  r3=0.05
        # II:  m3=0.01, r3=0.01S
        # III: m3=1.0,  r3=0.3

        self.model1 = ManiuplatorModel(Tp, 0.1, 0.05)
        self.model2 = ManiuplatorModel(Tp, 0.01, 0.01)
        self.model3 = ManiuplatorModel(Tp, 1.0, 0.3)


        self.models = [self.model1, self.model2, self.model3]
        self.i = 0

        self.Tp = Tp
        self.prev_u = np.zeros(2)
        self.prev_x = np.zeros(4)

        self.Kp = np.diag([25, 30])
        self.Kd = np.diag([30, 40])

    def choose_model(self, x):
        # TODO: Implement procedure of choosing the best fitting model from self.models (by setting self.i)

        """Wybiera model, który najlepiej pasuje do rzeczywistego x_dot."""

        # z poprzedniego symulacja i porównanie z aktualnym
        errors = []

        for model in self.models:

            x_dot = model.x_dot(self.prev_x, self.prev_u)
            #print(x_dot)
            x_mi = self.prev_x.reshape(4,1) + x_dot * self.Tp

            errors.append(np.sum(np.abs(x.reshape(4,1) - x_mi)))
            #print(x_mi)

        self.i = np.argmin(errors)


    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)

        isFLC = True

        q = x[:2]
        q_dot = x[2:]

        if isFLC:
            v = q_r_ddot + self.Kd @ (q_r_dot - q_dot) + self.Kp @ (q_r - q)
        else:
            v = q_r_ddot

        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v[:, np.newaxis] + C @ q_dot[:, np.newaxis]

        self.prev_u = u
        self.prev_x = x
        return u


