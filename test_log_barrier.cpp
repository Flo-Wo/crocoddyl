#include <boost/shared_ptr.hpp>

#include "crocoddyl/core/activations/log-barrier.hpp"

// g++ test_log_barrier.cpp -I/opt/homebrew/include -I/./include
void test_computation() {
  std::size_t nr = 1;
  Eigen::VectorXd weights_log_barrier = 1. * Eigen::VectorXd::Ones(nr);
  Eigen::VectorXd bound_log_barrier = 1.1 * Eigen::VectorXd::Ones(nr);
  double damping = 0.5;

  const boost::shared_ptr<crocoddyl::ActivationModelAbstract>& model =
      boost::make_shared<crocoddyl::ActivationModelLogBarrier>(
          weights_log_barrier, bound_log_barrier, damping);
  // create the corresponding data object
  boost::shared_ptr<crocoddyl::ActivationDataAbstract> data =
      model->createData();

  // Generating random input vector
  const Eigen::VectorXd& r = Eigen::VectorXd::Random(model->get_nr());

  std::cout << "Random input vector for the residual:" << std::endl;
  std::cout << r << std::endl;
  // data->a_value = nan("");

  // Getting the state dimension from calc() call
  model->calc(data, r);
  model->calcDiff(data, r);
  std::cout << "Value of the cost function:" << std::endl;
  std::cout << data->a_value << std::endl;
  std::cout << "Gradient of the objective:" << std::endl;
  std::cout << data->Ar << std::endl;
  std::cout << "Hessian of the objective:" << std::endl;
  std::cout << (data->Arr).diagonal() << std::endl;
}

int main(int argc, char** argv) {
  test_computation();
  return 0;
}