import numpy as np

#from models.free_model import FreeModel
from observers.eso import ESO
from .adrc_joint_controller import ADRCJointController
from .controller import Controller
#from models.ideal_model import IdealModel
from models.manipulator_model import ManiuplatorModel

class ADRFLController(Controller):
    def __init__(self, Tp, q0, Kp, Kd, p):

        self.model = ManiuplatorModel(Tp, 3, 0.05)

        self.Kp = Kp
        self.Kd = Kd

        p1 = p[0]
        p2 = p[1]

        self.L = np.array([
            [3 * p1, 0],
            [0, 3 * p2],
            [3 * p1 ** 2, 0],
            [0, 3 * p2 ** 2],
            [p1 ** 3, 0],
            [0, p2 ** 3]
        ])

        W = np.hstack([np.eye(2), np.zeros((2, 4))])

        A = np.zeros((6, 6))
        A[0, 2] = 1
        A[1, 3] = 1
        A[2, 4] = 1
        A[3, 5] = 1

        B = np.zeros((6,2))

        self.eso = ESO(A, B, W, self.L, q0, Tp)
        self.update_params(q0[:2], q0[2:])

    def update_params(self, q, q_dot):
        ### TODO Implement procedure to set eso.A and eso.B

        A = np.zeros((6, 6))
        A[0, 2] = 1
        A[1, 3] = 1
        A[2, 4] = 1
        A[3, 5] = 1

        x = np.concatenate([q, q_dot], axis=0)

        M_inv = np.linalg.inv(self.model.M(x))

        A[2:4,2:4] = - M_inv * self.model.C(x)


        B = np.zeros((6, 2))

        B[2:4, :] = M_inv


        self.eso.A = A
        self.eso.B = B

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        ### TODO implement centralized ADRFLC

        q = x[:2]

        z_hat = self.eso.get_state()
        q_hat = z_hat[0:2]
        q_dot_hat = z_hat[2:4]
        f_hat = z_hat[4:6]

        v = q_d_ddot + self.Kd @ (q_d_dot - q_dot_hat) + self.Kp @ (q_d - q_hat)


        # Wyznaczenie momentów sił τ
        u = self.model.M(x) @ (v - f_hat).reshape(2, 1)  + self.model.C(x) @ q_dot_hat.reshape(2, 1)
        u = u.flatten()

        self.update_params(q_hat, q_dot_hat)
        self.eso.update(q, u)

        return u


