#include "crocoddyl/core/activations/log-barrier.hpp"

#include "python/crocoddyl/core/activation-base.hpp"
#include "python/crocoddyl/core/core.hpp"
namespace crocoddyl {
namespace python {

void exposeActivationLogBarrier() {
  boost::python::register_ptr_to_python<
      boost::shared_ptr<ActivationModelLogBarrier> >();

  bp::class_<ActivationModelLogBarrier, bp::bases<ActivationModelAbstract> >(
      "ActivationModelLogBarrier", "Log Barrier function.\n\n",
      bp::init<Eigen::VectorXd, Eigen::VectorXd, bp::optional<double> >(
          bp::args("self", "weights", "bound", "damping"),
          "Initialize the activation model.\n\n"
          ":param weights: componentwise weights to linearly scale the "
          "residual vector"
          ":param bound: componentwise bound for the vector"
          ":param damping: Float to simulate a step function (use 1/t for "
          "large t)"))
      .def("calc", &ActivationModelLogBarrier::calc,
           bp::args("self", "data", "r"),
           "Compute log barrier loss\n\n"
           ":param data: activation data\n"
           ":param r: residual vector")
      .def("calcDiff", &ActivationModelLogBarrier::calcDiff,
           bp::args("self", "data", "r"),
           "Compute the derivatives of the log barrier function.\n\n"
           "It assumes that calc has been run first.\n"
           ":param data: activation data\n"
           ":param r: residual vector \n")
      .def("createData", &ActivationModelLogBarrier::createData,
           bp::args("self"), "Create the quadratic activation data.\n\n")
      .add_property("weights",
                    bp::make_function(&ActivationModelLogBarrier::get_weights,
                                      bp::return_internal_reference<>()),
                    &ActivationModelLogBarrier::set_weights,
                    "weights of the quadratic term");
}

}  // namespace python
}  // namespace crocoddyl