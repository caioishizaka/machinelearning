// Databricks notebook source
// MAGIC %md
// MAGIC # Custom transforms on a ML Pipeline
// MAGIC ### and how this helps dealing with NAs in an XGBoost model
// MAGIC 
// MAGIC Welcome to the series of 3 chapters (so far) on the ML journey using scala and pipelines.
// MAGIC 
// MAGIC By the end of this series you will:
// MAGIC 1. Have a good understanding on how a ML pipeline works, and how to use it
// MAGIC 2. Be able to create your own custom made transformers to use in your pipeline
// MAGIC 3. Understand how missing values (NAs) are treated in XGBoost, and how to properly deal with them
// MAGIC 
// MAGIC You may find out more about ML pipeline in this link: https://spark.apache.org/docs/latest/ml-pipeline.html#pipeline
// MAGIC 
// MAGIC The concept is rather simple. You create objects (called stages) that can be called on a dataset to produce a certain result. Yeah, crappy explanation, let's go to the specifics for clarity. There are 2 types of stages:
// MAGIC 
// MAGIC 1. Transformers: They literally transform a dataset, and return a new dataset with whatever transformations were specified
// MAGIC 2. Estimator: They will get a dataset, and create a predictive model out of it. The outcome is a ML model, that can be used to estimate a measure in new datasets
// MAGIC 
// MAGIC Why use ML pipeline?
// MAGIC 
// MAGIC I hope by the end of this series you realize they are super simple to use (once you get the grasp of it) and generates a very neat, streamlined final code. Instead of doing a lot of data manipulation outside of the model, then passing a final dataset to the model, you create all transformations, and those transformations are embedded in the model.
// MAGIC This is specially beneficial when applying the model to a new dataset. Instead of running a lot of code to prepare the data to be able to use model.fit(), the end result of a pipeline model will automatically run all those transformations.
// MAGIC 
// MAGIC Quick note on versions
// MAGIC 
// MAGIC For this I am using Databricks, on a cluster with 2 workers, using DBR 6.6. You should be able to yield the same results, as long as you are using spark 2.4.5 and scala 2.11
// MAGIC 
// MAGIC As for XGBoost, be careful. I will comment on different behaviors of the different versions, but for the bulk of this series I used [XGBoost4J 1.0.0](https://mvnrepository.com/artifact/ml.dmlc/xgboost4j_2.11/1.0.0). You will need both XGBoost4j and XGBoost4j-spark installed in the cluster for this to work. Should you choose Databricks own fork of that project, XGBoost-linux64, it is based on the 0.82 version of XGBoost4J. It is more stable for Databricks, but lacks significant features (such as calculating the gain of each variable)
// MAGIC 
// MAGIC All comments on versions will be about the XGBoost version, and will be referred simply as 1.0, 0.9 and 0.8, and the 0.8 will refer specifcally to XGBoost-linux64
// MAGIC 
// MAGIC Shall we begin?

// COMMAND ----------

// MAGIC %md
// MAGIC ##Chapter 1: Just getting our first ML pipeline working with out-of-shelf transformators and estimators
// MAGIC 
// MAGIC I decided to write this chapter because, unlike the common (and basically only) [tutorial on XGBoost](https://docs.databricks.com/applications/machine-learning/third-party/index.html#install-xgboost-on-databricks-runtime-ml) you can find for XGBoost4J, this will have both categorical and numerical variable as features, and this will lead to unexpected issues.
// MAGIC 
// MAGIC We need some data. So let's get it from: https://archive.ics.uci.edu/ml/datasets/Adult
// MAGIC 
// MAGIC I like it because it has more than a few thousand lines, a small (but not too small) number of features, numeric and categorical features, and a clear output (also categorical)
// MAGIC 
// MAGIC What I didn't like was how it handled NAs (with a '?'), and also the numeric variables didn't have any NAs. 
// MAGIC 
// MAGIC My manipulation was merely to make sure "?" would come in as Nulls, and also added 1% nulls to age, capital_gain and capital_loss

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
                     
display(data)

// COMMAND ----------

// MAGIC %md
// MAGIC As promised, no data manipulation whatsoever, we will leave those to the transformers down the road.
// MAGIC 
// MAGIC We will go straight to sampling (training and test) and on with transformations.

// COMMAND ----------

// DBTITLE 0,Sampling dataset
val Array(training, test) = data.randomSplit(Array(0.8, 0.2), seed = 123)

// COMMAND ----------

// MAGIC %md
// MAGIC Now we selected our data. Let's separate our categorical data (index_columns) from our numerical (numerical_columns), because we will treat them differently later on, applying different transformations. Also, let's keep our label (income) outside, as it does not belong to the feature space.
// MAGIC   
// MAGIC First decision we must make is choosing between education and education_number features. They are obviously 100% correlated, as education_number is just a integer representation of the factor. For simplicity of the example, I will use just education, but will discuss a little further when we treat it, and argue how education_number might produce better results.

// COMMAND ----------

val index_columns = Seq("workclass","education","marital_status","occupation","relationship","race","gender","native_country")
val numerical_columns = Seq("age","fnlwgt","capital_gain","capital_loss","hours_per_week")

// COMMAND ----------

// MAGIC %md
// MAGIC We are going to use XGBoost, which, like most ML models in scala, will take one vector column as feature, and one column as label.
// MAGIC 
// MAGIC In order to get the feature columns, I introduce our first transformer: VectorAssembler
// MAGIC 
// MAGIC It is super easy, setInputCols with an array with all the column names you want to consider as feature, and a setOutputCol with the name of the column you want it to generate
// MAGIC 
// MAGIC You can also setHandleInvalid, to determine what will happen to null/nan/missing values. If you choose "skip", the entire row will be removed if any of the feature columns is null/na/missing. If you choose "keep", it will not remove it.
// MAGIC 
// MAGIC For now, let's not worry about missing values, and just remove them

// COMMAND ----------

import org.apache.spark.ml.feature.{VectorAssembler} 

val simple_assembler = new VectorAssembler().setInputCols((index_columns ++ numerical_columns).toArray)
                                     .setOutputCol("features")
                                     .setHandleInvalid("skip")

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC Now let's take a look. Remember how I said a transformer will .transform() a dataset. Let's see how it does

// COMMAND ----------

try {
    display(simple_assembler.transform(training))
} 
catch {
    case error: Throwable => println("That didn't went well: \n" + error)
}


// COMMAND ----------

// MAGIC %md
// MAGIC Boom, it won't accept strings. So we need to convert the strings to something it will accept. That's what StringIndexer is for, to convert categorical variables into numbers (representing the category).
// MAGIC 
// MAGIC A column of values [A,B,B,A,C,D] would be converted to [0,1,1,0,2,3]. The StringIndexer will keep track of that match. It is also pretty smart. It will order the appearences, so the most recurrent category will be 0, then 1 and so forth.
// MAGIC 
// MAGIC As promised, let's discuss education_num again. Vector assembler will order the factors by appearence. That is probably great for most cases, the rarer the value, the higher the number, so it becomes easy to separate common values from rare ones. Nevertheless, it is not always the best relation. Say we have not that many people with just basic education, a lot of people with high education, and very few people with doctorate, for instance. Our assembler will put basic education closer to doctorate.
// MAGIC 
// MAGIC Therefore, if you have any hierarchy that makes sense beforehand, use it. I will leave this as to you, how would you consider education_num instead of education?
// MAGIC 
// MAGIC Back to the problem at hand. You will notice the StringIndexer options are very similar to VectorAssembler, which is great. Here we won't "skip" the missing values, as we will deal with them in the assembler anyway.
// MAGIC 
// MAGIC Quick trick, instead of creating an indexer for each column manually, I will leverage the index_columns array we defined previously to make this automatically

// COMMAND ----------

import org.apache.spark.ml.feature.{StringIndexer} 

val indexerArray = index_columns.map(column_name => {
  new StringIndexer()
    .setInputCol(column_name)
    .setOutputCol(s"${column_name}_index")
    .setHandleInvalid("keep")
    .fit(data)
}).toArray

// COMMAND ----------

// MAGIC %md
// MAGIC Let's first take a look at what those indexers do. Remember that they have a .transform() method, that will be return a Dataset. So we can nest all the StringIndexer.transform() to get the final result (which is exactly what the pipeline will do).
// MAGIC 
// MAGIC One quick note on versions. If you use 0.8, this will work even without the .fit() at the end. On 0.9 and 1.0, this will throw an error when you try to train the model. I don't know exactly why, but I have a guess on what .fit does. I believe it is used to asses which category will be mapped to each Int beforehand (to figure out the most occurring values). But I would have to take a look at the source code to be sure, and ain't nobody got time fo that (already had my fair share of source code reading writing this series)
// MAGIC 
// MAGIC If I am correct, than I have an additional advise. You will see on tutorials that the indexer is fitted for the training data. Which is fine in most cases, but in the rare occasion that you have a categorical value in your test set that is not present in your training data, I actually don't know what will happen, but it might give you some trouble. Also, if your training data has a different distribution than your overall data, the indices might differ a bit.
// MAGIC 
// MAGIC Not a lot of strong reasons not to fit it to training data, but exactly 0 reasons to to fit it to the whole data. Just use the entire data to fit the indexer.
// MAGIC 
// MAGIC Now back to our indexer, let's see it in action.

// COMMAND ----------

var temp_data = training
indexerArray.foreach{indexer =>
  temp_data = indexer.transform(temp_data)
}

display(temp_data)

// COMMAND ----------

// MAGIC %md
// MAGIC Pretty cool, right? Now all those strings are stored as integers. Super efficient storage. And it will make our assembler work

// COMMAND ----------

import org.apache.spark.ml.feature.{VectorAssembler} 

val indexed_assembler = new VectorAssembler().setInputCols((index_columns.map(name => s"${name}_index") ++ numerical_columns).toArray)
                                             .setOutputCol("features")
                                             .setHandleInvalid("skip")

display(indexed_assembler.transform(temp_data))

// COMMAND ----------

// MAGIC %md
// MAGIC Let's take a closer look at the feature column. You will see it is a vector with 4 elements. Let's explain those very quickly (as they will be key down the road)
// MAGIC 
// MAGIC 1. 0 or 1, indicating if this vector is Dense (1) or Sparse(0). A Dense vector will store all values of all columns. A Sparse vector will store the indices of the non-zero values and those values. This is done to save on memory and some performance gains. Remember how StringIndexer assigns 0 to the most recurrent values? This helps having as many 0s as possible, making Sparse vectors be more effective
// MAGIC 2. Size of the vector. Particularly important for Sparse vectors, as many fields may be omitted
// MAGIC 3. Indices list. Will be empty for Dense vectors (no point of indicating it as all values will be stored). Will have the indices of the non-zero values for Sparse vector
// MAGIC 4. The values. All of them for Dense vectors, the corresponding values of the indices for Sparse.
// MAGIC 
// MAGIC Wooh, that was a lot. But if you aim to understand what will happen when we introduce NAs to the model, it is good to understand this.
// MAGIC 
// MAGIC Last step before putting everything into the pipeline. We will create two last transformations. One that will convert our label (income) to an index, and one that will undo that afterwards. That way we can use XGBoost (or any other model) safely, and get back a label that makes sense (not some randomly mapped 0 or 1).

// COMMAND ----------

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

// COMMAND ----------

// MAGIC %md
// MAGIC *Now for the pipeline.*
// MAGIC 
// MAGIC As for the pipeline, look how easy it is. You define the stages in an Array, create a pipeline, set those stages, and you're good to go
// MAGIC 
// MAGIC Quick remark about versions:
// MAGIC 
// MAGIC 0.8: This will work normally, you can even omit the "allow_non_zero_for_missing" parameter
// MAGIC 0.9: You will need to add "missing" -> 0 parameter, and can omit "allow_non_zero_for_missing" parameter
// MAGIC 1.0: This will work normally, but you have to keep either the "allow_non_zero_for_missing" parameter or introduce "missing" -> 0
// MAGIC 
// MAGIC Don't worry quite so yet, as we will go into details in Chapter 3

// COMMAND ----------

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

val indexed_stages = indexerArray ++ Array(labelIndexer, indexed_assembler, xgbClassifier, labelConverter)
val indexed_pipeline = new Pipeline().setStages(indexed_stages)
val indexed_model = indexed_pipeline.fit(training)

// COMMAND ----------

// MAGIC %md
// MAGIC Okay, first win. Let's see how our model is doing
// MAGIC 
// MAGIC I won't go over different metrics, as it is not the scope here. I will stick to AUC (Area under ROC) which requires a simple column with (estimated probability, actual classification). If you are not familiarized with AUC, stop right now, as you shouldn't be worrying about ML pipeline transformations at the moment
// MAGIC 
// MAGIC But first, let's take a look at our results dataframe. And stop to appreciate how simple it is to apply the model on our test set. All the transformations are embedded, no manipulations required whatsoever

// COMMAND ----------

val results = indexed_model.transform(test)

display(results)

// COMMAND ----------

// MAGIC %md
// MAGIC Wow, it looks like it is failing a lot, what is wrong with it? I don't have an answer, it is just like it has flipped all predictions. I don't know why exactly it is doing it. If you have an answer, please tell me.
// MAGIC 
// MAGIC But in this kind of model I am more interested about the probabilities, and we can easily get that and use it as we want

// COMMAND ----------

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics  
import org.apache.spark.sql.functions.col

def getProb = udf((vector: org.apache.spark.ml.linalg.Vector) => vector.apply(0))

val predictionAndLabels = results.select(getProb(col("probability")),col("label")).as[(Double, Double)].rdd

val metrics = new BinaryClassificationMetrics(predictionAndLabels)
// val labels = metrics.labels

val auROC = metrics.areaUnderROC
println("Area under ROC = " + auROC)

// COMMAND ----------

// MAGIC %md
// MAGIC Not bad at all for our first ML model. In the next chapters we will go down the rabbit hole of data science and do a number of different things to improve the model (and most likely fail miserably)
// MAGIC 
// MAGIC Among the things we can do, we will explore 2 in the next chapters:
// MAGIC 
// MAGIC 1. Can normalizing numerical variables make a difference? For XGBoost, which is basically a boosted tree model, intuition says no, as normalizing won't impact the tree decisions. But it is a great way to learn how to build a custom transformer. Also this can be helpful for other (not tree-based) models, such as linear regression (specially with regularization)
// MAGIC 2. How can we make most of the data win Null/NaN/missing values? It is just 1% of missing values, but what if it were 30%? 80%? We couldn't afford filtering all out
