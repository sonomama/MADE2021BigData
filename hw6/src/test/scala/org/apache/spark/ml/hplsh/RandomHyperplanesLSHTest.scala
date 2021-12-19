package org.apache.spark.ml.hplsh

import org.scalatest._
import org.apache.spark.sql.DataFrame
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should
import org.apache.spark.ml.linalg
import org.apache.spark.ml.linalg.{Vector, Vectors}

class RandomHyperplanesLSHTest
  extends AnyFlatSpec with should.Matchers with WithSpark{
  lazy val data: DataFrame = HyperPlanesLSHTest.data
  lazy val planesNormals: Array[Vector] = HyperPlanesLSHTest.planesNormals
  val error = 1e-6

  "Model" should "transform data" in {
    val hyperPlanesLSH: HyperPlanesLSH = new HyperPlanesLSH()
      .setSeed(19)
      .setNumHashTables(4)
      .setInputCol("features")
      .setOutputCol("hashes")
    val model = hyperPlanesLSH.fit(data)
    val transformedData = model.transform(data)

    transformedData.count() should be(data.count())
  }

  "Model" should "return correct hash function" in {
    val HyperPlanesLSHModel: HyperPlanesLSHModel = new HyperPlanesLSHModel(planesNormals)
      .setInputCol("features")
      .setOutputCol("hashes")
    val testVector = linalg.Vectors.fromBreeze(breeze.linalg.Vector(2, 1, 4, -3))
    val sketch = HyperPlanesLSHModel.hashFunction(testVector)

    sketch.length should be(planesNormals.length)
    sketch(0)(0) should be(1.0)
    sketch(1)(0) should be(-1.0)
    sketch(2)(0) should be(1.0)
    sketch(3)(0) should be(1.0)
  }

  "Model" should "test hash distance" in {
    val HyperPlanesLSHModel: HyperPlanesLSHModel = new HyperPlanesLSHModel(planesNormals)
      .setInputCol("features")
      .setOutputCol("hashes")
    val testVector1 = linalg.Vectors.fromBreeze(breeze.linalg.Vector(-1, -2, 3, 4))
    val testVector2 = linalg.Vectors.fromBreeze(breeze.linalg.Vector(-3, 1, 1, 2))
    val sketch1 = HyperPlanesLSHModel.hashFunction(testVector1)
    val sketch2 = HyperPlanesLSHModel.hashFunction(testVector2)
    val similarity = HyperPlanesLSHModel.hashDistance(sketch1, sketch2)
    similarity should be (0.75 +- error)
  }

  object HyperPlanesLSHTest extends WithSpark {
    lazy val planesNormals = Array(
      Vectors.dense(1, 1, -1, -1),
      Vectors.dense(-1, 1, -1, 1),
      Vectors.dense(1, -1, 1, -1),
      Vectors.dense(-1, -1, 1, -1)
    )

    lazy val vectors = Seq(
      Vectors.dense(1, 2, 3, 4),
      Vectors.dense(-1, 2, 3, 5),
      Vectors.dense(1, -2, -3, 6)
    )

    lazy val data: DataFrame = {
      import sqlc.implicits._
      vectors.map(x => Tuple1(x)).toDF("features")
    }
  }
}
