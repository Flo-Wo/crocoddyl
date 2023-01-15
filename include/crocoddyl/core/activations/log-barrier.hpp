
#ifndef CROCODDYL_CORE_ACTIVATIONS_LOG_BARRIER_HPP_
#define CROCODDYL_CORE_ACTIVATIONS_LOG_BARRIER_HPP_

#include <iostream>
#include <stdexcept>

#include "crocoddyl/core/activation-base.hpp"
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

/**
 * @brief Logarithmic Barrier function
 *
 * The computation of the function and it derivatives are carried out in
 * `calc()` and `caldDiff()`, respectively.
 *
 * \sa `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ActivationModelLogBarrierTpl
    : public ActivationModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActivationModelAbstractTpl<Scalar> Base;
  typedef ActivationDataAbstractTpl<Scalar> ActivationDataAbstract;
  typedef ActivationDataLogBarrierTpl<Scalar> Data;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  explicit ActivationModelLogBarrierTpl(const VectorXs& weights,
                                        const VectorXs& bound,
                                        const Scalar damping = Scalar(1.0))
      : Base(weights.size()),
        weights_(weights),
        bound_(bound),
        damping_(damping){};
  virtual ~ActivationModelLogBarrierTpl(){};

  // define the computational methods
  virtual void calc(const boost::shared_ptr<ActivationDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& r) {
    if (static_cast<std::size_t>(r.size()) != nr_) {
      throw_pretty("Invalid argument: "
                   << "r has wrong dimension (it should be " +
                          std::to_string(nr_) + ")");
    }
    boost::shared_ptr<Data> d = boost::static_pointer_cast<Data>(data);

    data->a_value = Scalar(-1) * damping_ *
                    (bound_ - weights_.cwiseProduct(r)).array().log().sum();
  };

  /**
   * @brief Compute the derivatives of the smooth-abs function
   *
   * @param[in] data  Smooth-abs activation data
   * @param[in] r     Residual vector \f$\mathbf{r}\in\mathbb{R}^{nr}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<ActivationDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& r) {
    if (static_cast<std::size_t>(r.size()) != nr_) {
      throw_pretty("Invalid argument: "
                   << "r has wrong dimension (it should be " +
                          std::to_string(nr_) + ")");
    }

    boost::shared_ptr<Data> d = boost::static_pointer_cast<Data>(data);

    // computation of the gradient without the damping coefficient, given by
    // w/(b-w * x) componentwise -> save as intermediate result
    d->GradWithoutDamping = weights_.cwiseProduct(
        (bound_ - weights_.cwiseProduct(r)).array().inverse().matrix());

    data->Ar = damping_ * (d->GradWithoutDamping);

    // computation of the hessian, given by diag(w^2 \odot 1/(b-w*x)^2 )
    // the diagonal is just the pointwise squared gradient
    data->Arr.diagonal() =
        damping_ * (d->GradWithoutDamping).array().pow(2).matrix();
  };

  /**
   * @brief Create the smooth-abs activation data
   *
   * @return the activation data
   */
  virtual boost::shared_ptr<ActivationDataAbstract> createData() {
    return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
  };

  const VectorXs& get_weights() const { return weights_; };
  void set_weights(const VectorXs& weights) {
    if (weights.size() != weights_.size()) {
      throw_pretty("Invalid argument: "
                   << "weight vector has wrong dimension (it should be " +
                          std::to_string(weights_.size()) + ")");
    }

    weights_ = weights;
  };
  /**
   * @brief Print relevant information of the smooth-1norm model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const {
    os << "ActivationModelLogBarrier {nr=" << nr_ << "}";
  }

 protected:
  using Base::nr_;  //!< Dimension of the residual vector
 private:
  VectorXs weights_;  // componentwise weights
  VectorXs bound_;    //!< Componentwise upper bound
  Scalar damping_;  // factor of 1/damping in front of the log-barrier function
};

template <typename _Scalar>
struct ActivationDataLogBarrierTpl : public ActivationDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef ActivationDataAbstractTpl<Scalar> Base;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;

  template <typename Activation>
  explicit ActivationDataLogBarrierTpl(Activation* const activation)
      : Base(activation),
        GradWithoutDamping(VectorXs::Zero(activation->get_nr())) {}

  VectorXs GradWithoutDamping;  //!< Componentwise upper bound
  using Base::Arr;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTIVATIONS_SMOOTH_1NORM_HPP_
