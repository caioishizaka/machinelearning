// Databricks notebook source
// MAGIC %md
// MAGIC Previously...

// COMMAND ----------

import org.apache.spark.sql.functions.{col,when}
import org.apache.spark.sql.types.{IntegerType,StringType,StructType,StructField}
import org.apache.spark.SparkFiles

// spark.sparkContext.addFile("https://github.com/caioishizaka/machinelearning/raw/master/data/adult_data_treated.csv")
// dbutils.fs.mv("file:" + SparkFiles.get("adult_data_treated.csv"), "dbfs:/tmp/adult_data_treated.csv")

val adultSchema = StructType(Array(
  StructField("age", IntegerType, true),
  StructField("workclass", StringType, true),
  StructField("fnlwgt", IntegerType, true),
  StructField("education", StringType, true),
  StructField("education_num", IntegerType, true),
  StructField("marital_status", StringType, true),
  StructField("occupation", StringType, true),
  StructField("relationship", StringType, true),
  StructField("race", StringType, true),
  StructField("gender", StringType, true),
  StructField("capital_gain", IntegerType, true),
  StructField("capital_loss", IntegerType, true),
  StructField("hours_per_week", IntegerType, true),
  StructField("native_country", StringType, true),
  StructField("income", StringType, true)
))

val data = spark.read.format("csv")
                     .option("sep", ";")
                     .option("header", "false")
                     .schema(adultSchema)
                     .load("dbfs:/tmp/adult_data_treated.csv")

val Array(training, test) = data.randomSplit(Array(0.8, 0.2), seed = 123)

val index_columns = Seq("workclass","education","marital_status","occupation","relationship","race","gender","native_country")
val numerical_columns = Seq("age","fnlwgt","capital_gain","capital_loss","hours_per_week")

import org.apache.spark.ml.feature.{StringIndexer} 

val indexerArray = index_columns.map(column_name => {
  new StringIndexer()
    .setInputCol(column_name)
    .setOutputCol(s"${column_name}_index")
    .setHandleInvalid("keep")
    .fit(training)
}).toArray

import org.apache.spark.ml.feature.{IndexToString} 

val labelIndexer = new StringIndexer()
                            .setInputCol("income")
                            .setOutputCol("label")
                            .setHandleInvalid("skip") //no point keeping data we have no label on
                            .fit(data)

val labelConverter = new IndexToString()
                            .setInputCol("prediction")
                            .setOutputCol("predicted_income")
                            .setLabels(labelIndexer.labels)

import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassifier}
import org.apache.spark.ml.Pipeline

val xgbParam = Map("eta" -> 0.3,
      "allow_non_zero_for_missing" -> "true",             
      "max_depth" -> 3,
      "objective" -> "multi:softprob",
      "num_class" -> 2,       
      "num_round" -> 100,
      "num_workers" -> 3,
      "seed" -> 123)

val xgbClassifier = new XGBoostClassifier(xgbParam).setFeaturesCol("features").setLabelCol("label")

import org.apache.spark.sql.functions.{col,when,lit}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.util._
import org.apache.spark.ml.param.{Param, ParamMap, ParamValidators}
import org.apache.spark.ml.param.shared.{HasInputCol,HasOutputCol,HasHandleInvalid}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.{StructType,StructField,DoubleType}
import org.apache.spark.SparkException

class NumericNormalizer(override val uid: String) extends Transformer with HasInputCol with HasOutputCol with HasHandleInvalid{
  
  val mean: Param[Double] = new Param(this, "mean", """Mean of the column""")
  val stdDev: Param[Double] = new Param(this, "stdDev", """Standard Deviation of the column""")
  override val handleInvalid: Param[String]  = new Param[String](this, "handleInvalid",
    """ How to handle Null and Nan values
        keep: will convert Null to NaN
        skip: will filter out Null or NaN
    """, ParamValidators.inArray(Array("keep","skip")))
 
  setDefault(mean, 0.0)
  setDefault(stdDev, 1.0)
  
  def this() = this(Identifiable.randomUID("numNormalizer"))
  
  override def copy(extra: ParamMap): NumericNormalizer = defaultCopy(extra)
  
  def setOutputCol(value: String): this.type = set(outputCol, value)
      
  def setInputCol(value: String): this.type = set(inputCol, value)
  
  def setHandleInvalid(value: String): this.type = set(handleInvalid, value)
  
  def setMean(value: Double): this.type = set(mean, value)
  
  def setStdDev(value: Double): this.type = set(stdDev, value)
  
  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    
    var result = dataset
    ${handleInvalid} match {
      case "keep"  => result = result.withColumn(${outputCol},  when(col(${inputCol}).isNull, lit(Double.NaN)).otherwise(col(${inputCol}).cast("Double"))) // This will replace nulls by NaNs
      case "skip"  => result = result.withColumn(${outputCol},col(${inputCol})).filter(col(${inputCol}).isNotNull).filter(col(${inputCol}) === col(${inputCol})) //This will remove both Nulls and NaN
      case _ => throw new SparkException("""Error mode not supported, sorry""")
    }
    
    return result.withColumn(${outputCol}, (col(${outputCol}) - ${mean})/ ${stdDev})
  }
  
  override def transformSchema(schema: StructType): StructType = {StructType(schema.fields :+ new StructField(${outputCol}, DoubleType, true))}
}

import org.apache.spark.sql.functions.{mean, stddev}

val mean_map = numerical_columns.map(column => column -> data.agg(mean(column)).head().getDouble(0)).toMap
val stdDev_map = numerical_columns.map(column => column -> data.filter(col(column) < 500*mean_map(column)).agg(stddev(column)).head().getDouble(0)).toMap

val normalizerArray = numerical_columns.map(column_name => {
  new NumericNormalizer()
    .setInputCol(column_name)
    .setOutputCol(s"${column_name}_norm")
    .setHandleInvalid("keep")
    .setMean(mean_map(column_name))
    .setStdDev(stdDev_map(column_name))
}).toArray

// COMMAND ----------

// MAGIC %md
// MAGIC ##Chapter 3 - Dealing with NAs and XGBoost
// MAGIC 
// MAGIC Now we are made promises that XGBoost deals with NAs by itself. https://xgboost.readthedocs.io/en/latest/faq.html#how-to-deal-with-missing-value
// MAGIC 
// MAGIC Though it is true, XGBoost4J-Spark implementation is not 100% satisfactory in that sense. Here we have to discuss the different versions extensively, as they vary greatly on this regard. There is very few documentation on this, so most of my knowledge comes from forum discussions and source code reading.
// MAGIC 
// MAGIC First let's talk about the role of missing values in an XGBoost tree. A tree will make a decision on a variable. If it is categorical, it will choose which categories go to each leaf. If it is numerical, it will determine a threshold, numbers under the threshold go left, numbers above threshold goes right.
// MAGIC 
// MAGIC ![Survival of passengers of Titanic Decision Tree Model](https://upload.wikimedia.org/wikipedia/commons/e/eb/Decision_Tree.jpg)
// MAGIC 
// MAGIC Now, what role does missing values play in this decision? For categorical variables it should be pretty simple, it is another category, no big deal. For numerical variables it is a bit different. A missing value is neither above or below a threshold (nor equal). So it could go either way. XGBoost deals with it the proper way, it decides where it should go (based on data). If a missing value shows up in test but was not present in training, it will default to some of the nodes (left, if I am not mistaken).
// MAGIC 
// MAGIC Okay, you are now an expert on decision trees. Let's go to XGBoost4j now. The following code will work (i.e., it will run and produce results) in 1.0, but will fail in both 0.8 and 0.9

// COMMAND ----------

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{VectorAssembler}

val na_assembler = new VectorAssembler().setInputCols((index_columns.map(name => s"${name}_index") ++ numerical_columns.map(name => s"${name}_norm")).toArray)
                                             .setOutputCol("features")
                                             .setHandleInvalid("keep")

val na_stages = indexerArray ++ normalizerArray ++ Array(labelIndexer, na_assembler, xgbClassifier, labelConverter)
val na_pipeline = new Pipeline().setStages(na_stages)

val na_model = if (ml.dmlc.xgboost4j.scala.spark.VERSION == "1.0.0") na_pipeline.fit(training)

// COMMAND ----------

// MAGIC %md
// MAGIC Let's talk about what is going on. Missing values are handled very similarly in all versions, so let's start with the basic behavior. Here is what the process looks like:
// MAGIC 
// MAGIC 1. XGBoost gets the data
// MAGIC 2. If there are sparse vectors, missing values are considered, well, missing
// MAGIC 3. Additionally, any values in the vectors that match the "missing" parameter will also be considered missing
// MAGIC 
// MAGIC On version 0.9 they added a validation step
// MAGIC 
// MAGIC 4. If "missing" is different than 0, you have to have "allow_non_zero_for_missing" to be true, otherwise it will error
// MAGIC 
// MAGIC Fun fact, it is impossible to make 0.9 work without setting "missing" -> 0. Even if you pass the "allow_non_zero_for_missing_value" -> true, it will fail. I don't know what they did or how this made to release (there is a [specific test for that](https://github.com/dmlc/xgboost/blob/master/jvm-packages/xgboost4j-spark/src/test/scala/ml/dmlc/xgboost4j/scala/spark/MissingValueHandlingSuite.scala) in the repo), but it is true, I tested with Databricks 6.6 ML (which includes XGBoost4j 0.9 by default). Maybe a different distribution of 0.9 has that fixed.
// MAGIC 
// MAGIC Another fun fact, from version 0.9 to 1.0 they renamed "allow_non_zero_for_missing_value" to "allow_non_zero_for_missing", so for version 0.9, remember that.
// MAGIC 
// MAGIC Enough of (not so) fun facts, and back to the algorithm. Remember from chapter 1 how VectorAssembler can generate sparse vectors? Well, by default it will omit 0 as value. But that does not mean that there won't be dense vectors with 0s. It will decide whether to use dense or sparse by itself.
// MAGIC 
// MAGIC Now comes the problem. Missing values will be treated as missing, but they are in reality 0. Also, there are 0s in the Dense vectors, which will be treated as zeroes unless you set "missing" -> 0. Therefore you are in great risk of having some 0s (represented in sparse vectors) treated as missing and some 0s (in dense vectors) treated as 0s. For example, these 2 identical vectors (from VectorAssembler):
// MAGIC 
// MAGIC - [0, 3, [0,1], [1,1]] -> third feature will be treated as missing
// MAGIC - [1, 3, [1,1,0]]      -> third feature will be treated as zero
// MAGIC 
// MAGIC By default, we have "missing" -> NaN and "allow_non_zero_for_missing" -> false. In version 0.8, we don't have (4) check, so it will succeed, but it will treat the same original value differently, depending on the representation. A huge flaw. By setting "missing" -> 0, though it is not 100% correct to treat it as missing, at least you are consistently treating all 0s as missing.
// MAGIC 
// MAGIC All in all, I believe the additional check (4) is a good one. The only thing I would change is the default "missing" value. It is set to NaN by default, I would put 0 as default. If you don't specify "missing" and you have sparse vectors, it will fail. Make something with default parameters fail seems weird to me.
// MAGIC 
// MAGIC #### So, what to do?
// MAGIC 
// MAGIC Finally, some practical advice:
// MAGIC 
// MAGIC 1. For 0.8 and 0.9, don't sweat that much. Just set "missing" -> 0, or your model will either fail or produce wrong results. If you have real 0s and real NAs in the dataset, your model will fail anyway and you will need one of the following approaches
// MAGIC 
// MAGIC 2. For 1.0 just keep in mind what you are doing. If you are using VectorAssembler, keep in mind it is omitting 0s, so the only way to have a consistent model is to have "missing" -> 0. Nevertheless, if you have both 0s and NaN, the model will fail.
// MAGIC 
// MAGIC Now you may ask, how can I treat 0s and missing values correctly? Great question, let me go through 2 different paths. Inspiration comes from [their tutorial](https://xgboost.readthedocs.io/en/latest/jvm/xgboost4j_spark_tutorial.html#dealing-with-missing-values), which gives us 3 options
// MAGIC 
// MAGIC 1. Converting all vectors to Dense. Looks easy, we create a new transformer (which we are expert at doing right now) after the assembler, to convert all feature vectors to DenseVector. Nevertheless if you have a lot of NAs, it will be very inneficient on memory. If you don't have memory constraints, it is an okay path to go. I briefly tested it, with poor results. It will work on 1.0, but it will fail on 0.8, and to make it work on 0.9 you will have to set "missing" -> 0, so you need to replace your actual 0 values by something else (0.001 maybe?). I will not pursue this option here, but rather focus on the other 2
// MAGIC 
// MAGIC 2. Converting missing values to something else upstream. Very pragmatic approach. Just convert all the nulls/nans to something else, so you don't have to deal with them. I will show this method and discuss some shortcomings
// MAGIC 
// MAGIC 3. Replace the assembler by a custom transformer, so we can indicate the sparse value ourselves, instead of getting 0 for sparsity. This can be way more storage efficient than (1), depending on how many missing values you have.

// COMMAND ----------

// MAGIC %md
// MAGIC ### Replacing missing values by something else.
// MAGIC 
// MAGIC We don't have to worry about categorical variables. For StringIndexer, null is just another category, and will be treated as so, no confusion.
// MAGIC 
// MAGIC Risk is on Numerical Values. Remember how we made the NumericNormalizer replace missing values with NaN? Well, we could go back and make it replace it with something else, and leave it as a param for the user to decide. As we are normalizing for the mean, we can leave the default replacement for 0.0, which equals to the mean. I will comment further about replacing missing values by the mean later on, but as of now, it is a value as any other.
// MAGIC 
// MAGIC Key risk here is that we have actual 0s in our set. As we are normalizing, it is very unlikely to have a lot of values that are equal to the mean. Even in low cardinality columns, you have to be pretty unluck to have a numerical value to be exactly the mean. But feel free to examine your data beforehand. If you rather set it to -999999, feel free. But understand the consequences:
// MAGIC 
// MAGIC -999999 is a number. In versions 0.8 and 1.0 you can set the missing value as -999999, but if you are using VectorAssembler, remember it may omit a few 0s (if they exist), so you end up with the same issue described above. 
// MAGIC 
// MAGIC If you have both 0s and missing values in your data, I strongly suggest you to replace the 0s with something else (0.0001), and set the NAs to 0. That way, your VectorAssembler will omit the 0s, and 0s in the Dense vectors will be correctly treated as missing as well.

// COMMAND ----------

import org.apache.spark.ml.param.{Param, ParamMap, ParamValidators}
import org.apache.spark.ml.param.shared.{HasInputCol,HasOutputCol,HasHandleInvalid}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.{StructType,StructField,DoubleType}
import org.apache.spark.SparkException

// Extending an existing class, so much easier
class NumericNormalizerWithReplacement extends NumericNormalizer {
  
  val replacementValue: Param[Double] = new Param(this, "replacementValue", """Value to replace Null and NaNs""")
  
  setDefault(replacementValue, 0.0)
  
  def setReplacementValue(value: Double): this.type = set(replacementValue, value)
  
  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    
    var result = dataset.withColumn(${outputCol}, (col(${inputCol}) - ${mean})/ ${stdDev})
    ${handleInvalid} match {
      case "keep"  => result = result.withColumn(${outputCol},  when(col(${outputCol}).isNull, lit(${replacementValue})).otherwise(col(${outputCol}).cast("Double"))) //Now replacing Nulls with replacementValue
      case "skip"  => result = result.filter(col(${outputCol}).isNotNull).filter(col(${outputCol}) === col(${outputCol})) //This will remove both Nulls and NaN
      case "error" => throw new SparkException("""Error mode not supported, sorry""")
    }
    return result 
  }
}

// COMMAND ----------

val normalizerWithReplacementArray = numerical_columns.map(column_name => {
  new NumericNormalizerWithReplacement()
    .setInputCol(column_name)
    .setOutputCol(s"${column_name}_norm")
    .setHandleInvalid("keep")
    .setMean(data.agg(mean(column_name)).head().getDouble(0))
    .setStdDev(data.agg(stddev(column_name)).head().getDouble(0))
})

var temp_data = training

normalizerWithReplacementArray.foreach{normalizer =>
  temp_data = normalizer.transform(temp_data)
}

display(temp_data)

// COMMAND ----------

// MAGIC %md
// MAGIC Looks like it worked, we have some 0s instead of NaNs. Let's see how the model trains

// COMMAND ----------

import org.apache.spark.ml.Pipeline

val normalizerWithReplacementArray = numerical_columns.map(column_name => {
  new NumericNormalizerWithReplacement()
    .setInputCol(column_name)
    .setOutputCol(s"${column_name}_norm")
    .setHandleInvalid("keep")
    .setMean(data.agg(mean(column_name)).head().getDouble(0))
    .setStdDev(data.agg(stddev(column_name)).head().getDouble(0))
})

val xgbParam = Map("eta" -> 0.3,
//       "allow_non_zero_for_missing" -> "true",  
      "missing"   -> 0,
      "max_depth" -> 3,
      "objective" -> "multi:softprob",
      "num_class" -> 2,       
      "num_round" -> 100,
      "num_workers" -> 3,
      "seed" -> 123)

val xgbClassifier = new XGBoostClassifier(xgbParam).setFeaturesCol("features").setLabelCol("label")

val na_2_stages = indexerArray ++ normalizerWithReplacementArray ++ Array(labelIndexer, na_assembler, xgbClassifier, labelConverter)
val na_2_pipeline = new Pipeline().setStages(na_2_stages)
val na_2_model = na_2_pipeline.fit(training)

// COMMAND ----------

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics  
import org.apache.spark.sql.functions.col

def getProb = udf((vector: org.apache.spark.ml.linalg.Vector) => vector.apply(0))

val results = na_2_model.transform(test)

val predictionAndLabels = results.select(getProb(col("probability")),col("label")).as[(Double, Double)].rdd

val metrics = new BinaryClassificationMetrics(predictionAndLabels)
// val labels = metrics.labels

val auROC = metrics.areaUnderROC
println("Area under ROC = " + auROC)

// COMMAND ----------

// MAGIC %md
// MAGIC Yay! It worked. It didn't help much on our AUC. Well, life is hard. At least we were able to not filter out data just because 1 feature is missing. And we were able to run in all versions

// COMMAND ----------

// MAGIC %md
// MAGIC #### Best Option - Features as custom Sparse Vector
// MAGIC 
// MAGIC High of hope on this one. Documentation says it is the best approach and I believe it. Not blindly, but it makes sense. If we take off NaNs out of the vectors, then they can't cause an error anymore. And we have plenty of evidence that it will work with sparse vectors.
// MAGIC 
// MAGIC The idea here is replacing the assembler entirely. Our brand new custom made assembler will return only sparse vectors, and it will omit only NaNs. As we are running the normalizer beforehand, it is guaranteed all null values are now NaNs. If you want to make it more generic (to consider both NaNs and nulls), you can change the vectorizeRow function as you will. You may even create a parameter to tell which value should be considered as missing.
// MAGIC 
// MAGIC Nevertheless, being very pragmatic, we will just grab the data, and put it into a SparseVector, omitting NaN values. Here's my approach:
// MAGIC 
// MAGIC *vectorizeRow* is where the magic happens. As we will iterate over all the rows, nothing better than make it take a row and return a SparseVector. The SparseVector constructor takes 3 inputs, the size of the vector, an Array with the indices of non-zero values, and an Array with those values. You can check on code that it is very straightforward, grab the Row, get it size, transform it into array, get the indices of not-NaN values, get those values, construct and return the Vector
// MAGIC 
// MAGIC *SparseVectorAssembler* is going to be very similar to VectorAssembler. So similar that I will just extend it and override the transform part (even the transformSchema can be the same). Crazy right? Not so much if you think that is just a small twist on original VectorAssembler.

// COMMAND ----------

import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.ml.param.shared.{HasInputCols,HasOutputCol}
import org.apache.spark.sql.functions.struct

def vectorizeRow(row: Row): org.apache.spark.ml.linalg.SparseVector = {
  val size = row.length
  val row_array = row.toSeq.asInstanceOf[Seq[Double]].toArray
  val indices = row_array.zipWithIndex.filter(!_._1.isNaN).map(_._2)
  val values = row_array.zipWithIndex.filter(!_._1.isNaN).map(_._1)
  return new SparseVector(size, indices, values)
}

def vectorizeRowUDF = udf(vectorizeRow _)

class SparseVectorAssembler extends VectorAssembler {
  
  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)

    return dataset.withColumn(${outputCol}, vectorizeRowUDF(struct(${inputCols}.map(col):_*)))
  }

}


// COMMAND ----------

val na_3_assembler = new SparseVectorAssembler().setInputCols((index_columns.map(name => s"${name}_index") ++ numerical_columns.map(name => s"${name}_norm")).toArray)
                                                .setOutputCol("features")
var temp = training

indexerArray.foreach {indexer =>
  temp = indexer.transform(temp)
}

normalizerArray.foreach {normalizer =>
  temp = normalizer.transform(temp)
}

temp = na_3_assembler.transform(temp)


display(temp)

// COMMAND ----------

// MAGIC %md
// MAGIC Boom! Sparse vectors for the win. No more NaN in our vectors. Lots of 0s, showing that it worked.
// MAGIC 
// MAGIC But will it work on the pipeline to train the model?
// MAGIC 
// MAGIC It won't work in 0.9 (sorry), but this will work for both 0.8 and 1.0. I am passing both "allow_non_zero_for_missing_value" and "allow_non_zero_for_missing" for compatibility, choose just the one that works for your version. Or keep both, it won't break a thing

// COMMAND ----------

import org.apache.spark.ml.Pipeline

val na_3_xgbParam = Map("eta" -> 0.3,
      "allow_non_zero_for_missing_value" -> true,
      "allow_non_zero_for_missing" -> true,
      "max_depth" -> 3,
      "objective" -> "multi:softprob",
      "num_class" -> 2,       
      "num_round" -> 100,
      "num_workers" -> 3)

val na_3_xgbClassifier = new XGBoostClassifier(na_3_xgbParam).setFeaturesCol("features").setLabelCol("label")

val na_3_stages = indexerArray ++ normalizerArray ++ Array(labelIndexer, na_3_assembler, na_3_xgbClassifier, labelConverter)
val na_3_pipeline = new Pipeline().setStages(na_3_stages)
val na_3_model = na_3_pipeline.fit(training)

// COMMAND ----------

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics  
import org.apache.spark.sql.functions.col

def getProb = udf((vector: org.apache.spark.ml.linalg.Vector) => vector.apply(0))

val results = na_3_model.transform(test)

val predictionAndLabels = results.select(getProb(col("probability")),col("label")).as[(Double, Double)].rdd

val metrics = new BinaryClassificationMetrics(predictionAndLabels)
// val labels = metrics.labels

val auROC = metrics.areaUnderROC
println("Area under ROC = " + auROC)

// COMMAND ----------

// MAGIC %md
// MAGIC It is alive! But no improvement in AUC. But that was not the point. ML is not black and white, the best approach will depend on the problem and the data. Instead of 1% NAs, what if it were 10%, 30%, 80%. Filtering out NAs is probably not going to yield better results.
// MAGIC 
// MAGIC You were given 2 approaches on dealing with NAs (instead of filtering them out). I would strongly recommend using the last one. As you saw, it is really easy to implement (no matter what the doucumentation said about extra work), and it treats NAs as what they are, missing values. Also it removes any confusion caused by sparse vectors motting 0s.
// MAGIC 
// MAGIC The other approach was replacing NAs by a value. Let me briefly talk about that as promised eariler.
// MAGIC 
// MAGIC I am a big fan of replacing NAs by the mean of the distribution. But this may skew the model when you have a lot of NAs. You can do something more sophisticated, replacing by N(m,s), which is a random number based on a normal distribution with mean m and std dev s. This would keep your distribution "unharmed". If it is normally distributed, of course. If you know more about your feature distribution beforehand, I highly recommend a tailored made approach (including to the normalization).
// MAGIC 
// MAGIC But you must always ask yourself: what does a missing value actually represents? Quick example, suppose you run a survey, and ask if the respondent owns a car. If they say yes, you ask how many cars. But if they say no, they won't be shown that answer. In a dataset, that may come as a missing value for all people that answered no the car question. Now, is it fair to fill the data with the mean of the answers? Definitely not, in this case it is pretty clear that those missing values should be replaced by zero.
// MAGIC 
// MAGIC Another case that it may be a problem is that when you have a lot of missing values. If you have 80% of missing values, for instance, filling them with the mean (with or without random noise) will certainly bias your model towards the mean. Adding the random noise won't help much as well. If the variance of data is just random, you won't be able to model it (as it is random). You may fool yourself, and overfit model, but in the end, it is just noise. Any good model will just tell you that the variables don't really matter, and return you the mean of your output as an answer. Every machine learning model is just trying to do the same thing, find correlation between features and output, to build a predictive model. And if you remember correctly from your statistics books, what is correlation but the covariance of the variables divided by the product of each standard deviation? In other words, it is just a measure if the variance of two variables seems to be coupled together. When one goes above the average, does the other go too? 
// MAGIC 
// MAGIC Therefore, when I said I am a big fan of replacing NAs by the mean, it is not without boundary conditions. For a small portion of NAs (like 1% in our example), it should be more than fine. For 80%+ of missing data, definitely don't do it. And you may ask, what kind of problem have so many missing values? A lot of them actually. As we navigate through a world of unstructured data, you will find a lot of holes in your dataset. Imagine which kind of features Google search algorithm uses to show YOU (specifically) the best results for a specific search query. Or the features FB uses to determine which is the best Ad to diplay to you. Imagine all the data it has for the people, and all the holes.
// MAGIC 
// MAGIC But let me finish here leaving you with the mother of sparse data problems. The one that was made famous in 2009. The Netflix Challenge. Giving the ratings every single user gave to each individual movie, what is the best estimate of the rating of all other movies (the user didn't rate) for each user? The core of Netflix recommendation system. I won't talk about the solution itself, but just the sparsity aspect of each. Millions of users, tens of thousands of movies. Even if we make a very generous assumption that the average user rates 100 movies/tv shows, it would still leave us with (at least) 99% of missing data. How would you feel about filling the gaps with the mean?
// MAGIC 
// MAGIC Hope this helped give you some clarity on how XGBoost4j works, how it treats missing values, and what is the best way to treat missing values yourself.
