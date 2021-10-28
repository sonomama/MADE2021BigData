import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object Main {
  val DATA_PATH = "../hw4/src/main/data/tripadvisor_hotel_reviews.csv"
  val TOPN = 100

  def getIdf(docsCount: Long, docFreq: Long): Double =
    math.log((docsCount.toDouble + 1) / (docFreq.toDouble + 1))

  def main(args : Array[String]): Unit ={
    val spark  = SparkSession.builder()
      .master("local[*]")
      .appName("tfidf")
      .getOrCreate()

    import spark.implicits._

    val df = spark.read.option("header", true)
      .csv(DATA_PATH)
    val df_cleaned = df.filter(df.col("Review").isNotNull)
      .withColumn("Review_id", monotonically_increasing_id())
      .withColumn("Review", lower(col("Review")))
      .withColumn("Review", trim($"Review"))
      .withColumn("Review", regexp_replace($"Review",
        "[^a-zA-Z0-9]+", " "))

    val docsCount = df_cleaned.count()

    // TF
    val reviews = df_cleaned
      .withColumn("Review", split(col("Review"), " "))
      .drop("Rating")
      .select($"Review_id", explode($"Review") as "token")
      .where(length($"token") > 1 or $"token".cast("int").isNotNull)

    val tf = reviews.groupBy("Review_id", "token")
      .agg(count("token") as "tf")

    // IDF
    val docf = reviews.groupBy("token")
      .agg(countDistinct("Review_id") as "df")
      .orderBy($"df".desc).limit(TOPN)
    val getIdfUdf = udf { df: Long => getIdf(docsCount, df) }
    val idf = docf.withColumn("idf", getIdfUdf(col("df")))

    val tfidf = tf.join(idf, Seq("token"), "inner")
      .withColumn("tf_idf", col("tf") * col("idf"))
      .select(col("Review_id"), col("token"), col("tf_idf"))

    // Pivotted table
    val pivot_result = tfidf.groupBy("Review_id")
      .pivot("token")
      .agg(round(
        first(col("tf_idf")), 3))
      .orderBy($"Review_id".asc)
    pivot_result.show()
  }
}
