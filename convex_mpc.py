import numpy as np
import utils
from params import *

class ConvexMpc():
    def __init__(self):
        pass
    
    def calculate_Rz(self, euler):
        """_summary_

        Args:
            euler (_type_): _description_

        Returns:
            _type_: _description_
        """
        cos_yaw = np.cos(euler[2])
        sin_yaw = np.sin(euler[2])
        cos_pitch = np.cos(euler[1])
        tan_pitch = np.tan(euler[1])
                
        Rz = np.array([
            [cos_yaw / cos_pitch, sin_yaw / cos_pitch, 0],
            [-sin_yaw, cos_yaw, 0],
            [cos_yaw * tan_pitch, sin_yaw * tan_pitch, 1]
        ]) #Transpose of Rz
        return Rz
        
    def calculate_A_mat(self, Rz):
        """_summary_

        Args:
            Rz (_type_): _description_

        Returns:
            _type_: _description_
        """
        O3 = np.zeros((3,3))
        I3 = np.eye(3)
        
        A = np.block([
            [O3,O3,Rz,O3],
            [O3,O3,O3,I3],
            [O3,O3,O3,O3],
            [O3,O3,O3,O3]
        ])
        return A
        
    def calculate_B_mat(self, robot_mass, trunk_inertia, Rz,leg_pose):
        """_summary_

        Args:
            robot_mass (_type_): _description_
            trunk_inertia (_type_): _description_
            Rz (_type_): _description_
            leg_pose (_type_): _description_

        Returns:
            _type_: _description_
        """
        I = Rz.T@trunk_inertia@Rz
        row = []
        for i in range(4): #for each leg
            O3 = np.zeros((3,3))
            I3 = np.eye(3)
            row.append(np.array([
                [O3],
                [O3],
                [np.linalg.inv(I)@utils.skew(leg_pose[i])],
                [I3/robot_mass]
            ]))
        B = np.block([row[1],row[2],row[3],row[4]])
        return B
    
    def calculate_qp_mat(self,euler):
        """calculate A_qp and B_qp
        standard QP formulation:
        minimize 1/2 * xT * H * x + q' * x subject to lb <= C * x <= ub
        H: hessian
        q: gradient
        C: linear constraints
        
        A_qp = [A,
                A^2,
                A^3,
                ...
                A^k]'

        B_qp = [A^0*B(0),
                A^1*B(0),     B(1),
                A^2*B(0),     A*B(1),       B(2),
                ...
                A^(k-1)*B(0), A^(k-2)*B(1), A^(k-3)*B(2), ... B(k-1)]

        Args:
            euler (_type_): _description_

        Returns:
            _type_: _description_
        """
        # create empty block matrix https://stackoverflow.com/questions/58118451/how-to-index-whole-matrix-from-block-with-python
        
        B_qp = np.zeros((PLAN_HORIZON**2,PLAN_HORIZON**2))  # shape (9,9)
        A_qp = np.zeros((PLAN_HORIZON,1))
        Rz = self.calculate_Rz(euler)
        A0 = self.calculate_A_mat(Rz)
        for i in range(PLAN_HORIZON):
            if i == 0:
                A_qp[i] = A0
            else:
                A_qp[i] = A_qp[i-1]@A0
            for j in range(i+1):
                if i == j:
                    B_qp[i*3:i*3+3,j*3:j*3+3] = mat
                else:
                    B_qp[i*3:i*3+3,j*3:j*3+3] = mat
        B_qp = np.array([])
        return A_qp, B_qp, H, C
            
            
    