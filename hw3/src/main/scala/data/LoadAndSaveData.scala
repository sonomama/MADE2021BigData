package data
import breeze.linalg._
import java.io._


object LoadAndSaveData {

  def loadCSV(file_path: String): DenseMatrix[Double] = {
    val data = csvread(new File(file_path), ',', skipLines = 1)
    data
  }

  def saveMetrics(cvError: Double, valError: Double, filePath: String): Unit = {
    val file = new File(filePath)
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(s"CV MSE: ${cvError}, validation set MSE: ${valError}")
    bw.close()
  }
}
