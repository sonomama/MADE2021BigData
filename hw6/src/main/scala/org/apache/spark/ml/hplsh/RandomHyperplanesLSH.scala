package org.apache.spark.ml.hplsh

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.linalg
import org.apache.spark.ml.feature.{LSH, LSHModel}
import org.apache.spark.ml.linalg.{Matrices, Matrix, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol, HasSeed}
import org.apache.spark.ml.util.{DefaultParamsReader, DefaultParamsWriter,
  Identifiable, MLReadable, MLReader, MLWriter, SchemaUtils}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.StructType

import scala.util.Random


trait HyperPlanesLSHParams extends HasInputCol with HasOutputCol {

  def setInputCol(value: String) : this.type = set(inputCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)
  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getInputCol, new VectorUDT())
    if (schema.fieldNames.contains($(outputCol))) {
      SchemaUtils.checkColumnType(schema, getOutputCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getInputCol).copy(name = getOutputCol))
    }
  }
}

class HyperPlanesLSH (override val uid: String)
  extends LSH[HyperPlanesLSHModel] with HasSeed {

  override def setInputCol(value: String): this.type = super.setInputCol(value)
  override def setOutputCol(value: String): this.type = super.setOutputCol(value)
  override def setNumHashTables(value: Int): this.type = super.setNumHashTables(value)
  def setSeed(value: Long): this.type = set(seed, value)
  def this() = this(Identifiable.randomUID("rhp-lsh"))
  override protected[this] def createRawLSHModel(inputDim: Int): HyperPlanesLSHModel = {
    val rng = new Random($(seed))
    val planesNormals = Array.fill($(numHashTables)) {
      val randArray = Array.fill(inputDim)(
        {if (rng.nextGaussian() > 0) 1.0 else -1.0}
      )
      linalg.Vectors.fromBreeze(breeze.linalg.Vector(randArray))
    }
    new HyperPlanesLSHModel(uid, planesNormals)
  }
  override def copy(extra: ParamMap): this.type = defaultCopy(extra)
  override def transformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, $(inputCol), new VectorUDT)
    validateAndTransformSchema(schema)
  }
}


class HyperPlanesLSHModel private[hplsh](override val uid: String,
                                         private[hplsh] val planesNormals: Array[Vector]
                                         )
  extends LSHModel[HyperPlanesLSHModel] {
  override def setInputCol(value: String): this.type = super.set(inputCol, value)
  override def setOutputCol(value: String): this.type = super.set(outputCol, value)
  private[hplsh] def this(planeNormals: Array[Vector]) =
    this(Identifiable.randomUID("rhp-lsh"), planeNormals)
  override protected[ml] def hashFunction(elems: linalg.Vector): Array[linalg.Vector] = {
    val hashValues: Array[Int] = planesNormals.map(
      planeNormal => if (elems.dot(planeNormal) > 0) 1 else -1
    )
    hashValues.map(Vectors.dense(_))
  }

  override protected[ml] def keyDistance(x: linalg.Vector,
                                         y: linalg.Vector): Double = {
    Math.sqrt(Vectors.sqdist(x, y))
  }

  override protected[ml] def hashDistance(x: Array[linalg.Vector],
                                          y: Array[linalg.Vector]): Double = {
    x.zip(y).map(pair => if (pair._1 == pair._2) 1 else 0).sum.toDouble / x.size
  }

  override def write: MLWriter = {
    new HyperPlanesLSHModel.HyperPlanesLSHModelWriter(this)
  }

  override def copy(extra: ParamMap): HyperPlanesLSHModel = {
    val copied = new HyperPlanesLSHModel(uid, planesNormals).setParent(parent)
    copyValues(copied, extra)
  }
}


object HyperPlanesLSHModel extends MLReadable[HyperPlanesLSHModel] {
  override def load(path: String): HyperPlanesLSHModel = super.load(path)

  private[HyperPlanesLSHModel] class HyperPlanesLSHModelWriter(instance: HyperPlanesLSHModel)
    extends MLWriter {
    private case class Data(planesNormals: Matrix)
    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      val numRows = instance.planesNormals.length
      val numCols = instance.planesNormals.head.size
      val values: Array[Double] = instance.planesNormals.map(_.toArray).reduce(Array.concat(_, _))
      val randMatrix: Matrix = Matrices.dense(numRows, numCols, values)
      val data = Data(randMatrix)
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(Seq(data))
        .repartition(1)
        .write.parquet(dataPath)
    }
  }

  override def read: MLReader[HyperPlanesLSHModel] = new MLReader[HyperPlanesLSHModel] {
    private val className = classOf[HyperPlanesLSHModel].getName
    override def load(path: String): HyperPlanesLSHModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val dataPath = new Path(path, "data").toString
      val data = sparkSession.read.parquet(dataPath)
      val Row(planesNormals: Matrix) = MLUtils.convertMatrixColumnsToML(data,
        "planesNormals")
        .select("planesNormals").head()
      val model = new HyperPlanesLSHModel(metadata.uid,
        planesNormals.rowIter.toArray)
      metadata.getAndSetParams(model)
      model
    }
  }
}