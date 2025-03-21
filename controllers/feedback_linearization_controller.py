import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManiuplatorModel(Tp)

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Implementacja feedback linearization: wyznacz momenty sił τ, aby układ zachowywał się jak układ liniowy.
        """
        print(q_r_dot)
        q = x[:2]  # Aktualna pozycja (q1, q2)
        q_dot = x[2:]  # Aktualna prędkość (q̇1, q̇2)

        # Obliczenie macierzy M, C na podstawie aktualnego stanu
        M = self.model.M(x)
        C = self.model.C(x)

        # Współczynniki PD (dostosuj je!)
        #Kp = np.diag([0.4, 0.4])  # Przykładowe wartości dla PD
        #Kd = np.diag([0.1, 0.1])

        # Definicja sygnału sterującego v z PD
        #v = q_r_ddot + Kd @ (q_dot - q_r_dot) + Kp @ (q - q_r)
        v = q_r_ddot
        # Wyznaczenie momentów sił τ
        tau = M @ v + C @ q_dot

        return tau
