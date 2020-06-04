// Databricks notebook source
// MAGIC %md
// MAGIC Previously...

// COMMAND ----------

// DBTITLE 0,Getting data
import org.apache.spark.sql.functions.{col,when}
import org.apache.spark.sql.types.{IntegerType,StringType,StructType,StructField}
import org.apache.spark.SparkFiles

spark.sparkContext.addFile("https://github.com/caioishizaka/machinelearning/raw/master/data/adult_data_treated.csv")
dbutils.fs.mv("file:" + SparkFiles.get("adult_data_treated.csv"), "dbfs:/tmp/adult_data_treated.csv")

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
      "num_workers" -> 3)//,
//       "seed" -> 123)

val xgbClassifier = new XGBoostClassifier(xgbParam).setFeaturesCol("features").setLabelCol("label")

// COMMAND ----------

// MAGIC %md
// MAGIC ## Chapter 2 - Custom transformations
// MAGIC 
// MAGIC As promised, let's make our first custom transformation. This will be very simple. You will pass which column you want to be normalized, the mean and the standard deviation to be considered, and the other params we are used to (handleInvalid and outputCol). We will use standard score, which is the easiest to implement and will achieve what we need in most cases. https://en.wikipedia.org/wiki/Standard_score
// MAGIC 
// MAGIC This will basically do
// MAGIC 
// MAGIC $$
// MAGIC z = \frac{X - \mu}{\sigma}
// MAGIC $$
// MAGIC 
// MAGIC Where \mu is the mean and \sigma is the standard deviation of the population.
// MAGIC 
// MAGIC Now I can't quite break the next cell in parts, so I will explain it thoroughly here.
// MAGIC 
// MAGIC #### Basic needs
// MAGIC In order for your transformer to work it need a few things.
// MAGIC 1. transform() method, as it will be called by the pipeline. It has to return a Dataset/Dataframe
// MAGIC 2. transformSchema() method. I don't quite understand why. My best guess is that it will assess the schema of the result beforehand, to understand if there are any conflicts in the pipeline before starting transforming (and figuring out in the last step it is broken because a column is missing)
// MAGIC 3. It has to extend Transformer. I am not 100% sure about this, but this is definitely the easiest way to achieve a lot of the needs.
// MAGIC 4. It needs an Identifiable. Also not sure why, but it does. Line 23 below takes care of that. Also, initializing it with an uid.
// MAGIC 5. copy method. I am just using the default one (line 25)
// MAGIC 
// MAGIC Lots of things, but the bulk of the work will be at (1), a little bit at (2) if necessary. And some work getting params right. Let's go right to them
// MAGIC 
// MAGIC #### How params work
// MAGIC I would be lying if I told you I know. As I said before, I went down a rabbit hole to make NAs work with XGBoost, ended up in the source code of Assemblers and Pipeline, and emerged with a way to do it.
// MAGIC 
// MAGIC So what I can tell is, params are a nice way to store information, cannot be accessed from the outside (unless you have a method for that), and can be easily managed via methods.
// MAGIC 
// MAGIC What is the difference from an attribute? I honestly don't know. They are different in nature and properties, but at a fundamental level I couldn't tell you why use one or the other. I am just going with the flow, those transformers are usually implemented with Params, let's mimic them
// MAGIC 
// MAGIC Now for the real deal. We can extend many classes with those params, such as HasInputCol, HasOutputCol and HasHandleInvalid. This will create the params for us (and some methods as well). But we can create our own params from scratch.
// MAGIC 
// MAGIC We will create the params mean and stdDev. This happen in lines 12 and 13. On lines 14-18 we override handleInvalid, basically because it was giving me trouble. Very simple syntax, new Param(parent, name, documentation)
// MAGIC 
// MAGIC To access a param inside the class you use ${ParamName}
// MAGIC 
// MAGIC To create a method to set a param, it is also easy, as you can see on lines 27-35
// MAGIC 
// MAGIC #### Why don't we calculate the mean and std dev ourselves
// MAGIC Great question. While 100% possible, it has a fundamental flaw. Mean and Standard Deviation are depending on the data. If your test data has a different distribution than your training data, values will be mapped differently. On the most extreme situation, when you pass a dataset with one element, you will not have any comparison to normalize.
// MAGIC 
// MAGIC By giving fixed numbers (based on your training set, your whole set, whatever you want) you avoid that pitfall.
// MAGIC 
// MAGIC #### The transform
// MAGIC Now that we know all the requirements let's go down to implementation. As you can see, our transform method does 2 things, treats null/nan values accordingly to the handleInvalid parameter, and applies the standard score, creating a new column in the process
// MAGIC Not rocket science here. But this is beautiful. All the magic happens in this method, and the sky is the limit. Just make sure you don't do anything that is not serializable (crappy vague advice, I know, but I also don't quite understand what is and what is not serializable)
// MAGIC 
// MAGIC Quick note on replacing nulls by NaN on the "keep" method. This is will have great impact in Chapter 3. As of now, just do it.
// MAGIC 
// MAGIC #### The transformSchema
// MAGIC Key thing, call it on .transform(). Why? I don't know, everyone does it, and if you don't it will fail.
// MAGIC 
// MAGIC Now, this is super easy. You should be able to know what is going to be the resulting schema based on the dataset schema that is coming in. Just write it on words.

// COMMAND ----------

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

// COMMAND ----------

// MAGIC %md
// MAGIC That was a lot. Let's see if it is working properly with a single column, age

// COMMAND ----------

import org.apache.spark.sql.functions.{mean, stddev}

val age_mean = data.agg(mean("age")).head().getDouble(0)
val age_stdDev = data.agg(stddev("age")).head().getDouble(0)

val ageNormalizer = new NumericNormalizer()
    .setInputCol("age")
    .setOutputCol("age_norm")
    .setHandleInvalid("keep")
    .setMean(age_mean)
    .setStdDev(age_stdDev)

display(ageNormalizer.transform(data).select("age","age_norm"))

// COMMAND ----------

// MAGIC %md
// MAGIC Looks like success to me. Null being mapped to NaN. Normalized values around 0.
// MAGIC Now let's create normalizer for all numerical data, and put in the pipeline. We will create a map of mean and standard deviations as well, to makes things clearer. Also, I have a feeling if you just make the calculation inside the normalizer setMean and setStdDev params, it will calculate it every time (not confirmed, it is just a guess)
// MAGIC 
// MAGIC Again, you have an option to use only training data or entire data to calculate the mean and standard deviation. I prefer to use the entire data to avoid any skew/bias the sampling may have caused. Though unlikely, it may happen. Also, I am confident that just knowing the mean and standard deviation of the entire set won't cause any bias/overfitting on the model itself.

// COMMAND ----------

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{VectorAssembler}


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

val normalized_assembler = new VectorAssembler().setInputCols((index_columns.map(name => s"${name}_index") ++ numerical_columns.map(name => s"${name}_norm")).toArray)
                                                .setOutputCol("features")
                                                .setHandleInvalid("skip")

val normalized_stages = indexerArray ++ normalizerArray ++ Array(labelIndexer, normalized_assembler, xgbClassifier, labelConverter)
val normalized_pipeline = new Pipeline().setStages(normalized_stages)
val normalized_model = normalized_pipeline.fit(training)

// COMMAND ----------

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics  
import org.apache.spark.sql.functions.col

def getProb = udf((vector: org.apache.spark.ml.linalg.Vector) => vector.apply(0))

val results = normalized_model.transform(test)

val predictionAndLabels = results.select(getProb(col("probability")),col("label")).as[(Double, Double)].rdd

val metrics = new BinaryClassificationMetrics(predictionAndLabels)
// val labels = metrics.labels

val auROC = metrics.areaUnderROC
println("Area under ROC = " + auROC)

// COMMAND ----------

// MAGIC %md
// MAGIC Almost the same thing, as intuition told you so.
// MAGIC 
// MAGIC But you learned how to create a transformer (not the giant robot cars unfortunately), isn't that cool? You can use it to anything. Removing outliers, grouping categorical values, making fun feature interaction, applying a function (log, exp, x^2, sin, ...) to a feature/label, making sense of some text. You can do anything.
// MAGIC 
// MAGIC A quick note on normalizing. Though close to useless on XGBoost tree model, it is very important on other models, such as linear and logistic regression, specially if you are using regularization. Regularization is nothing but keeping the coefficients small to avoid overfitting. If your data is not normalized, your coefficients may vary in magnitude. It is very simple to realize that. Imagine you have a model:
// MAGIC 
// MAGIC $$ Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 $$
// MAGIC 
// MAGIC It is very easy to see that if you multiply all X_1 by 100, the resulting model is going to be
// MAGIC 
// MAGIC $$ Y = \beta_0 + \beta_1 X_1 + \frac{\beta_2}{100} X_2 $$
// MAGIC 
// MAGIC Now if you regularize that, guess which coefficient is going to suffer more?
// MAGIC 
// MAGIC Normalizing deals with that and greatly improve the efficacy of regularization methods
