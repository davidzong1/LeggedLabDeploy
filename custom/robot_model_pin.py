from pinocchio import casadi as cpin
import pinocchio as pin
import numpy as np


class robot_dynamics:

    def __init__(self, dynamics_name: str, urdf_path: str):
        self._model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self._data = self._model.createData()
        self.q = np.array(self._model.nq)
        self.v = np.array(self._model.nv)
        self.last_v = np.array(self._model.nv)
        self.rot_mat = np.eye(3)
        self.control_dt = 0.02
        self.update_kin = False
        self.M = None
        self.C = None
        self.G = None
        self.CMM = None
        self.dCMM = None
        self.CM = None
        self.dCM = None
        self.CoM = None

    def model(self):
        return self._model

    @property
    def data(self):
        if not self.update_kin:
            self.forwardKinematics()
        return self._data

    def update_state(self, base_pos, base_vel, joint_pos, joint_vel, dt):
        self.rot_mat = pin.Quaternion(base_pos[3:7]).toRotationMatrix()
        self.q = np.concatenate((base_pos, joint_pos))
        self.v = np.concatenate((base_vel, joint_vel))
        self.control_dt = dt
        self.acc = (self.v - self.last_v) / self.control_dt
        self.last_v = self.v.copy()
        self.update_kin = False

    def forwardKinematics(self):
        pin.forwardKinematics(self._model, self._data, self.q, self.v, self.acc)
        pin.updateFramePlacements(self._model, self._data)
        self.update_kin = True

    def WB_dynamics(self):
        if not self.update_kin:
            self.forwardKinematics()
        # 质心动量矩阵(A矩阵)
        self.M = pin.crba(self._model, self._data, self.q)
        self.C = pin.nonLinearEffects(self._model, self._data, self.q, self.v)
        self.G = pin.computeGeneralizedGravity(self._model, self._data, self.q)

    # com dynamic Algorithm
    def COM_dynamic(self):
        if not self.update_kin:
            self.forwardKinematics()
        # 质心动量矩阵(A矩阵)
        self.CMM = pin.computeCentroidalMap(self._model, self._data, self.q)
        self.dCMM = pin.computeCentroidalMapTimeVariation(
            self._model, self._data, self.q, self.v
        )
        # 基座系下的质心动量
        self.CM = pin.computeCentroidalMomentum(self._model, self._data, self.q, self.v)
        self.dCM = pin.computeCentroidalMomentumTimeVariation(
            self._model, self._data, self.q, self.v, self.acc
        )
        # 世界坐标系下的质心位置
        self.CoM = pin.centerOfMass(self._model, self._data, self.q)
