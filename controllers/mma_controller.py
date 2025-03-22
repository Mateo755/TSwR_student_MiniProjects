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

    def choose_model(self, x):
        # TODO: Implement procedure of choosing the best fitting model from self.models (by setting self.i)

        """Wybiera model, który najlepiej pasuje do rzeczywistego x_dot."""
        min_error = float('inf')
        for i, model in enumerate(self.models):
            # Przewidywanie stanu z modelu

            x_dot = model.x_dot(x, self.prev_u)

            # Zaktualizowanie stanu o krok czasowy Tp
            x_mi = x + x_dot * self.Tp  # Integracja numeryczna (metoda Eulera)

            # Obliczanie błędu (różnica między rzeczywistym stanem a przewidywanym stanem)
            error = np.linalg.norm(x - x_mi)

            if error < min_error:
                min_error = error
                self.i = i



    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)

        q = x[:2]
        q_dot = x[2:]

        Kp = np.diag([25, 30])
        Kd = np.diag([30, 40])

        v = q_r_ddot + Kd @ (q_r_dot - q_dot) + Kp @ (q_r - q)
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v[:, np.newaxis] + C @ q_dot[:, np.newaxis]

        self.prev_u = u
        return u


