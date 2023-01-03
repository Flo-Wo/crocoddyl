
#ifndef CROCODDYL_CORE_ACTIVATIONS_LOG_BARRIER_HPP_
#define CROCODDYL_CORE_ACTIVATIONS_LOG_BARRIER_HPP_

#include <iostream>
#include <stdexcept>

#include "crocoddyl/core/activation-base.hpp"
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

/**
 * @brief Smooth-abs activation
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
  typedef ActivationDataLogBarrier<Scalar> Data;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  explicit ActivationModelLogBarrierTpl(const std::size_t nr,
                                        const Scalar bound = Scalar(1.))
      : Base(nr), bound_(bound){};
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

    // TODO: maybe save the inner value to avoid a recomputation in calcDiff
    d->a = bound_ - r.squaredNorm();
    data->a_value = -Scalar(0.5) * log(d->a);
    // data->a_value = -Scalar(0.5) * log(bound_ - r.squaredNorm());
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
    // computation of the gradient
    data->Ar = -r / (d->a);

    // computation of the hessian
    data->Arr = 1 / ((d->a) * (d->a)) * r * r.transpose();
    data->Arr.diagonal() += -1 / (d->a);

    // data->Ar = r.cwiseProduct(d->a.cwiseInverse());
    // data->Arr.diagonal() =
    //     d->a.cwiseProduct(d->a).cwiseProduct(d->a).cwiseInverse();
  };

  /**
   * @brief Create the smooth-abs activation data
   *
   * @return the activation data
   */
  virtual boost::shared_ptr<ActivationDataAbstract> createData() {
    return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
  };

  /**
   * @brief Print relevant information of the smooth-1norm model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const {
    os << "ActivationModelLogBarrier {nr=" << nr_ << ", bound=" << bound_
       << "}";
  }

 protected:
  using Base::nr_;  //!< Dimension of the residual vector
  Scalar bound_;    //!< Smoothing factor
};

template <typename _Scalar>
struct ActivationDataLogBarrierTpl : public ActivationDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef ActivationDataAbstractTpl<Scalar> Base;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <typename Activation>
  explicit ActivationDataLogBarrierTpl(Activation* const activation)
      : Base(activation), a(0) {}

  Scalar a;
  using Base::Arr;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTIVATIONS_SMOOTH_1NORM_HPP_
