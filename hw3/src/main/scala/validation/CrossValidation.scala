package validation

import breeze.linalg._
import scala.util.Random
import model.LinearRegression

object Validation {

  def MSE(y1: Array[Double], y2: Array[Double]): Double = {
    (y1 - y2).map(x => x * x).sum / y1.size
  }

  def crossValidation(k_folds: Int = 5, model: LinearRegression,
                      data: DenseMatrix[Double], y: Array[Double]): Double = {
    // shuffle indices and slice data and target y
    val shuffled_indices = Random.shuffle((0 to data.rows - 1).toList).toSeq
    val X = data(shuffled_indices, ::)
    val Y = for (i <-0 until shuffled_indices.length) yield y(shuffled_indices(i))

    // prepare vars for folds processing
    val len: Int = data.rows / k_folds
    var start: Int = 0
    var end: Int = len - 1
    var folds = 0

    var errors: List[Double] = List[Double]()
    while (folds != k_folds) {
      // choose indices for all except the particular fold
      val indices: IndexedSeq[Int] = (0 to start - 1) ++ (end + 1 to X.rows - 1)
      model.fit(new DenseMatrix(rows=indices.length,
                                cols=X.cols,
                                data=X(indices, ::).toDenseMatrix.toArray),
        (Y.slice(0, start) ++ Y.slice(end + 1, X.rows)).toArray, lr=1e-6)

      val yPred = model.predict(new DenseMatrix(rows=end - start + 1,
        cols=X.cols,
        data=X(start to end, ::).toDenseMatrix.toArray))

      val yTarget = Y.slice(start, end + 1).toArray
      val meanSqError = MSE(yTarget, yPred)
      errors = errors :+ meanSqError

      start += len
      if (folds != k_folds - 1) {
        end += len
      } else {
        end = data.rows
      }
      folds += 1
    }

    errors.sum / errors.size
    }
}
