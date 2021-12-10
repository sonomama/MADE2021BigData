package org.apache.spark.ml.customlinreg

import com.google.common.io.Files
import breeze.linalg.{*, DenseMatrix, DenseVector}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.DataFrame
import org.scalatest.flatspec._
import org.scalatest.matchers.should


class LinRegTest extends AnyFlatSpec with should.Matchers with WithSpark{
  val dataSize = 100
  val dataParams = DenseVector[Double](1.5, 0.3, -0.7)
  val paramDelta = 1e-3
  val mseThreshold = 1e-6
  val featColName = "x"
  val targetColName = "y"
  val predColName = "prediction"

  lazy val X = DenseMatrix.rand[Double](dataSize, dataParams.size)
  lazy val y = X * dataParams + DenseVector.rand[Double](dataSize) * 0.001
  lazy val df = generateDF(X, y)

  private def generateDF(X: DenseMatrix[Double], y: DenseVector[Double]): DataFrame = {
    import sqlc.implicits._
    lazy val data: DenseMatrix[Double] = DenseMatrix.horzcat(X, y.asDenseMatrix.t)
    lazy val _df = data(*, ::).iterator
      .map(x => (x(0), x(1), x(2), x(3)))
      .toSeq.toDF("x1", "x2", "x3", "y")

    lazy val assembler = new VectorAssembler()
      .setInputCols(Array("x1", "x2", "x3")).setOutputCol("x")
    lazy val df = assembler.transform(_df).select("x", "y")
    df
  }

  private def validateModel(model: LinRegModel): Unit = {
    val dfWithPrediction = model.transform(df)
    val evaluator = new RegressionEvaluator()
      .setLabelCol("y")
      .setPredictionCol("prediction")
      .setMetricName("mse")
    val mse = evaluator.evaluate(dfWithPrediction)
    mse should be <= mseThreshold
  }

  "Estimator" should s"have params: $dataParams" in {
    val lr = new LinReg()
      .setInputCol(featColName)
      .setOutputCol(targetColName)
    val model = lr.fit(df)
    val params = model.getWeights
    params(0) should be(dataParams(0) +- paramDelta)
    params(1) should be(dataParams(1) +- paramDelta)
    params(2) should be(dataParams(2) +- paramDelta)
  }

  "Model" should "have MSE < threshold" in {
    val lr = new LinReg()
      .setInputCol(featColName)
      .setOutputCol(targetColName)
      .setPredictionCol(predColName)
    val model = lr.fit(df)
    validateModel(model)
  }

  "Estimator" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(
      Array(
        new LinReg()
          .setInputCol(featColName)
          .setOutputCol(targetColName)
          .setPredictionCol(predColName)
      )
    )
    val tmpFolder = Files.createTempDir()
    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)
    val reRead = Pipeline
      .load(tmpFolder.getAbsolutePath)
      .fit(df)
      .stages(0)
      .asInstanceOf[LinRegModel]
    validateModel(reRead)
  }

  "Model" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(
      Array(
        new LinReg()
          .setInputCol(featColName)
          .setOutputCol(targetColName)
          .setPredictionCol(predColName)
      )
    )
    val model = pipeline.fit(df)
    val tmpFolder = Files.createTempDir()
    model.write.overwrite().save(tmpFolder.getAbsolutePath)
    val reRead = PipelineModel.load(tmpFolder.getAbsolutePath)
    validateModel(reRead.stages(0).asInstanceOf[LinRegModel])
  }
}
