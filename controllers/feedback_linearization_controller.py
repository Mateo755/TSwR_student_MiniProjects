import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManiuplatorModel(Tp, 3, 0.05)

        self.Kp = np.diag([25, 30])
        self.Kd = np.diag([30, 40])


    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Implementacja feedback linearization: wyznacz momenty sił τ, aby układ zachowywał się jak układ liniowy.
        """
        isfeedbackPD = True


        q = x[:2]  # Aktualna pozycja (q1, q2)
        q_dot = x[2:]  # Aktualna prędkość (q̇1, q̇2)


        # Definicja sygnału sterującego v z PD
        if isfeedbackPD:
            v = q_r_ddot + self.Kd @ (q_r_dot - q_dot) + self.Kp @ (q_r - q)
        else:
            v = q_r_ddot

        # Wyznaczenie momentów sił τ
        tau = self.model.M(x) @ v + self.model.C(x) @ q_dot

        return tau
