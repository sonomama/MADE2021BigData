import breeze.linalg.{DenseMatrix, _}
import com.typesafe.scalalogging._
import model.LinearRegression
import data.LoadAndSaveData._
import validation.Validation._


object Main {
  val CROSSVALSIZE = 0.7
  val KFOLDS = 2

  val logger = Logger("ScalaProject")

  def main(args: Array[String]): Unit = {
    if (args.length == 0) {
      logger.error("No input data")
      throw new Exception("No file path provided")
    }
    
    val file_path: String = args(0)
    val data: DenseMatrix[Double] = loadCSV(file_path)
    logger.info("Input data processed")

    val X: DenseMatrix[Double] = data(::, 1 to 2)
    val y: Array[Double] = data(::, 3).toArray
    val model = new LinearRegression()

    val error = crossValidation(KFOLDS, model, X, y)
    logger.info(s"Cross validation done with mean MSE value: ${error}")

    val XTrain: DenseMatrix[Double] = X(0 to (CROSSVALSIZE * X.rows).toInt - 1, ::)
    val yTrain = y.slice(0, (CROSSVALSIZE * X.rows).toInt)
    val XTest = X((CROSSVALSIZE * X.rows).toInt to X.rows - 1, ::)
    val yTest = y.slice((CROSSVALSIZE * X.rows).toInt, X.rows)

    model.fit(XTrain, yTrain)
    logger.info("Finished model fitting")
    saveMetrics(error, MSE(yTest, model.predict(XTest)), args(1))
    logger.info("Metrics saved")
  }
}

