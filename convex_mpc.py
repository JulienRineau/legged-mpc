import numpy as np
import utils
from params import *
import osqp
import numpy as np
import scipy as sp
from scipy import sparse

class ConvexMpc():
    def __init__(self):
        pass
    
    def calculate_Rz(self, euler):
        """_summary_

        Args:
            euler (_type_): _description_
        """
        cos_yaw = np.cos(euler[2])
        sin_yaw = np.sin(euler[2])
        cos_pitch = np.cos(euler[1])
        tan_pitch = np.tan(euler[1])
                
        self.Rz = np.array([
            [cos_yaw / cos_pitch, sin_yaw / cos_pitch, 0],
            [-sin_yaw, cos_yaw, 0],
            [cos_yaw * tan_pitch, sin_yaw * tan_pitch, 1]
        ]) #Transpose of Rz
        
    def calculate_A_mat(self, Rz):
        """_summary_

        Args:
            Rz (_type_): _description_

        Returns:
            _type_: _description_
        """
        O3 = np.zeros((3,3))
        I3 = np.eye(3)
        
        self.A_mat = np.block([
            [O3,O3,Rz,O3],
            [O3,O3,O3,I3],
            [O3,O3,O3,O3],
            [O3,O3,O3,O3]
        ])
        
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
        self.B_mat = np.block([row[1],row[2],row[3],row[4]])
    
    def calculate_qp_mat(self, euler):
        """calculate A_qp and B_qp
        standard QP formulation:
        minimize 1/2 * xT * P * x + qT * x subject to lb <= C * x <= ub
        P: hessian
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
        return A_qp, B_qp, P, C
    
    # TODO: translate funtion under for isaac gym
    
    def osqp_solve(self, x_current, x_ref):
        # Discrete time model of a quadcopter
        Ad = sparse.csc_matrix(calculate_A_mat(self.Rz))
        Bd = sparse.csc_matrix(calculate_B_mat(ROBOT_MASS, self.trunk_inertia, self.Rz, leg_pose))
        [nx, nu] = Bd.shape

        # Constraints
        u0 = 10.5916
        umin = np.array([9.6, 9.6, 9.6, 9.6]) - u0
        umax = np.array([13., 13., 13., 13.]) - u0
        xmin = np.array([-np.pi/6,-np.pi/6,-np.inf,-np.inf,-np.inf,-1.,
                        -np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf])
        xmax = np.array([ np.pi/6, np.pi/6, np.inf, np.inf, np.inf, np.inf,
                        np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])

        # Objective function
        Q = sparse.diags([0., 0., 10., 10., 10., 10., 0., 0., 0., 5., 5., 5.])
        QN = Q
        R = 0.1*sparse.eye(4)

        # Initial and reference states
        x0 = x_current #np.array 1x12
        xr = x_ref

        # Prediction horizon
        N = PLAN_HORIZON

        # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
        # - quadratic objective
        P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                            sparse.kron(sparse.eye(N), R)], format='csc')
        # - linear objective
        q = np.hstack([np.kron(np.ones(N), -Q.dot(xr)), -QN.dot(xr),
                    np.zeros(N*nu)])
        # - linear dynamics
        Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad)
        Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd)
        Aeq = sparse.hstack([Ax, Bu])
        leq = np.hstack([-x0, np.zeros(N*nx)])
        ueq = leq
        # - input and state constraints
        Aineq = sparse.eye((N+1)*nx + N*nu)
        lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
        uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])
        # - OSQP constraints
        A = sparse.vstack([Aeq, Aineq], format='csc')
        l = np.hstack([leq, lineq])
        u = np.hstack([ueq, uineq])

        # Create an OSQP object
        prob = osqp.OSQP()

        # Setup workspace
        prob.setup(P, q, A, l, u, warm_start=True)

        # Simulate in closed loop
        nsim = 15
        for i in range(nsim):
            # Solve
            res = prob.solve()

            # Check solver status
            if res.info.status != 'solved':
                raise ValueError('OSQP did not solve the problem!')

            # Apply first control input to the plant
            ctrl = res.x[-N*nu:-(N-1)*nu]
            x0 = Ad.dot(x0) + Bd.dot(ctrl)

            # Update initial state
            l[:nx] = -x0
            u[:nx] = -x0
            prob.update(l=l, u=u)

    def compute_contact_force(self,
                            desired_acc,
                            contacts,
                            acc_weight=ACC_WEIGHT,
                            reg_weight=1e-4,
                            friction_coef=0.45,
                            f_min_ratio=0.1,
                            f_max_ratio=10.):
        mass_matrix = compute_mass_matrix(
            robot.MPC_BODY_MASS,
            np.array(robot.MPC_BODY_INERTIA).reshape((3, 3)),
            robot.GetFootPositionsInBaseFrame())
        G, a = compute_objective_matrix(mass_matrix, desired_acc, acc_weight,
                                        reg_weight)
        C, b = compute_constraint_matrix(robot.MPC_BODY_MASS, contacts,
                                        friction_coef, f_min_ratio, f_max_ratio)
        G += 1e-4 * np.eye(12)
        result = quadprog.solve_qp(G, a, C, b)
        return -result[0].reshape((4, 3))

    def compute_constraint_matrix(mpc_body_mass,
                            contacts,
                            friction_coef=0.8,
                            f_min_ratio=0.1,
                            f_max_ratio=10):
        f_min = f_min_ratio * mpc_body_mass * 9.8
        f_max = f_max_ratio * mpc_body_mass * 9.8
        A = np.zeros((24, 12))
        lb = np.zeros(24)
        for leg_id in range(4):
            A[leg_id * 2, leg_id * 3 + 2] = 1
            A[leg_id * 2 + 1, leg_id * 3 + 2] = -1
            if contacts[leg_id]:
            lb[leg_id * 2], lb[leg_id * 2 + 1] = f_min, -f_max
            else:
            lb[leg_id * 2] = -1e-7
            lb[leg_id * 2 + 1] = -1e-7

        # Friction constraints
        for leg_id in range(4):
            row_id = 8 + leg_id * 4
            col_id = leg_id * 3
            lb[row_id:row_id + 4] = np.array([0, 0, 0, 0])
            A[row_id, col_id:col_id + 3] = np.array([1, 0, friction_coef])
            A[row_id + 1, col_id:col_id + 3] = np.array([-1, 0, friction_coef])
            A[row_id + 2, col_id:col_id + 3] = np.array([0, 1, friction_coef])
            A[row_id + 3, col_id:col_id + 3] = np.array([0, -1, friction_coef])
        return A.T, lb
                
            
    