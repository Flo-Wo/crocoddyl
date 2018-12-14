from crocoddyL.costs.cost import RunningQuadraticCostData
from crocoddyL.costs.cost import RunningQuadraticCost
import pinocchio as se3
import numpy as np


class StateRunningData(RunningQuadraticCostData):
  def __init__(self, dynamicModel):
    # Creating the data structure for a running quadratic cost
    RunningQuadraticCostData.__init__(self, dynamicModel, dynamicModel.nx())

    # Creating the data for the desired state
    self.x_des = np.zeros((dynamicModel.nxImpl(),1))


class ControlRunningData(RunningQuadraticCostData):
  def __init__(self, dynamicModel):
    # Creating the data structure for a running quadratic cost
    RunningQuadraticCostData.__init__(self, dynamicModel, dynamicModel.nu())

    # Creating the data for the desired control
    self.u_des = np.zeros((dynamicModel.nu(),1))


class StateCost(RunningQuadraticCost):
  """ State tracking cost.

  An important remark here is that the residual depends linearly on the state.
  So, for efficiency, we overwrite the updateQuadraticAppr function because we
  don't need 1) to find a linear approximation of the residual function, 2) to
  update the terms related to the control, and 3) to update the Hessian w.r.t
  the state (constant value).
  """
  def __init__(self, dynamicModel, weights):
    RunningQuadraticCost.__init__(self, dynamicModel.nx(), weights)
    self.dynamicModel = dynamicModel

  def createData(self, dynamicModel):
    data = StateRunningData(dynamicModel)
    np.copyto(data.rx, np.identity(dynamicModel.nx()))
    np.copyto(data.lxx, np.diag(self.weight.reshape(-1)))
    return data

  def updateResidual(self, costData, dynamicData, x, u):
    np.copyto(costData.r,
      self.dynamicModel.differenceState(dynamicData, costData.x_des, x))

  def updateResidualLinearAppr(self, costData, dynamicData, x, u):
    # Due to the residual is equals to x, we don't need to linearize each time.
    # So, rx is defined during the construction.
    return

  def updateQuadraticAppr(self, costData, dynamicData, x, u):
    # We overwrite this function since this residual is equals to x. So, rx is
    # a vector of 1s, and it's not needed to multiple them.

    # Updating the linear approximation of the residual function
    self.updateResidualLinearAppr(costData, dynamicData, x, u)

    # Updating the quadratic approximation of the cost function. We don't
    # overwrite again the lxx since is constant. This value is defined during
    # the construction. Additionally the gradient and Hession of the cost w.r.t.
    # the control remains zero.
    np.copyto(costData.lx, np.multiply(self.weight, costData.r))

  @staticmethod
  def setReference(costData, x_des):
    np.copyto(costData.x_des, x_des)


class ControlCost(RunningQuadraticCost):
  """ Control tracking cost.

  An important remark here is that the residual depends linearly on the control.
  So, for efficiency, we overwrite the updateQuadraticAppr function because we
  don't need 1) to find a linear approximation of the residual function, 2) to
  update the terms related to the state, and 3) to update the Hessian w.r.t
  the control (constant value).
  """
  def __init__(self, dynamicModel, weights):
    RunningQuadraticCost.__init__(self, dynamicModel.nu(), weights)

  def createData(self, dynamicModel):
    data = ControlRunningData(dynamicModel)
    np.copyto(data.ru, np.identity(dynamicModel.nu()))
    np.copyto(data.luu, np.diag(self.weight.reshape(-1)))
    return data

  def updateResidual(self, costData, dynamicData, x, u):
    np.copyto(costData.r, u - costData.u_des)

  def updateResidualLinearAppr(self, costData, dynamicData, x, u):
    # Due to the residual is equals to u, we don't need to linearize each time.
    # So, ru is defined during the construction.
    return

  def updateQuadraticAppr(self, costData, dynamicData, x, u):
    # We overwrite this function since this residual is equals to u. So, ru is
    # a vector of 1s, and it's not needed to multiple them.

    # Updating the linear approximation of the residual function
    self.updateResidualLinearAppr(costData, dynamicData, x, u)

    # Updating the quadratic approximation of the cost function. We don't
    # overwrite again the luu since is constant. This value is defined during
    # the construction. Additionally the gradient and Hession of the cost w.r.t.
    # the state remains zero.
    np.copyto(costData.lu, np.multiply(self.weight, costData.r))

  @staticmethod
  def setReference(costData, u_des):
    np.copyto(costData.u_des, u_des)