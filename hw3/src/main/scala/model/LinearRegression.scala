package model

import breeze.linalg._

class LinearRegression() {
  var thetas = new DenseMatrix(0, 0, Array.empty[Double])

  def fit(X: DenseMatrix[Double], Y: Array[Double], lr: Double = 1e-5): Unit = {
    thetas = DenseMatrix.rand[Double](X.cols + 1, 1)
    val ones = new DenseMatrix(X.rows, 1, Array.fill[Double](X.rows)(1))
    val Y_ = new DenseMatrix(X.rows, 1, Y)
    val X_ = DenseMatrix.horzcat(X, ones)
    var thetas_prev: DenseMatrix[Double] = new DenseMatrix(X_.cols, 1,
      Array.fill[Double](X_.cols)(Double.PositiveInfinity))

    while (functions.euclideanDistance(thetas.toDenseVector,
                                       thetas_prev.toDenseVector) > 1e-7) {
      thetas_prev = thetas.copy
      val dthetas: DenseMatrix[Double] = 2.0 / X_.rows *
        (X_.t * (X_ * thetas - Y_))
      thetas -= dthetas * lr
    }
  }

  def predict(X: DenseMatrix[Double]): Array[Double] = {
     (
      X * thetas(0 to thetas.rows - 2, ::) +
      new DenseMatrix(
        rows=X.rows, cols=1,
          data=Array.fill[Double](X.rows)(elem=thetas(thetas.rows - 1, 0))
        )
      ).toArray
  }
}
