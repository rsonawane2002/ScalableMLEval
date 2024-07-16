import sys
import time

import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

#these are the configurations for our executor_cores/ instances to see how our runtimes/efficiency changes
configurations = [["2", "2"], ["20", "4"], ["50", "6"], ["100", "8"]]

#these are the configurations for the number of threads we run on 
config_parallel = ["1", "5", "50", "100", "500"]

#these are the configurations for how we shuffle and repartition our dataframe
repartitions = [2, 5, 25, 50, 100]
runtimes = []
accuracies = []

for i in range(1):  

    #instances = config[0]
    #num_cores = config[1]

    start_time = time.time()

    # Step 1: Create a SparkSession
    spark = SparkSession.builder.appName("Random Forest Example").getOrCreate()
    #spark = SparkSession.builder.appName("Random Forest Example").config("spark.executor.instances", instances).config("spark.executor.cores", num_cores).getOrCreate()
    #spark = SparkSession.builder.appName("Random Forest Example").config("spark.default.parallelism", config).getOrCreate()
    
    file_name = "iris.csv"
    feature_cols = ["sepallength", "sepalwidth", "petallength", "petalwidth"]
    label_col = "label"
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
        feature_cols = ["subject", "condition", "replication", "leg", "joint", "time"]
        label_col = "angle"

    # Step 2: Load the dataset into PySpark DataFrame
    df = spark.read.csv(file_name, header=True, inferSchema=True)

    #df = df.repartition(config)

    # Step 3: Prepare the features for model training
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df = assembler.transform(df)

    # Convert the label to numbers for the random forest to work
    if file_name == "iris.csv":
        indexer = StringIndexer(inputCol="variety", outputCol="label")
        df = indexer.fit(df).transform(df)

        # Step 4: Initialize Random Forest Classifier
        rf = RandomForestClassifier(featuresCol="features", labelCol=label_col, numTrees=20)

        # Define evaluator
        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

    else:
        rf = RandomForestRegressor(featuresCol="features", labelCol=label_col, numTrees=20)

        # Define evaluator
        evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="r2")

    # Define parameter grid for cross-validation
    paramGrid = ParamGridBuilder().build()

    # Initialize CrossValidator to use cross validation to evaluate model
    crossval = CrossValidator(estimator=rf,
                            estimatorParamMaps=paramGrid,
                            evaluator=evaluator,
                            numFolds=10)
    
    avg_value = 0

    # Run cross-validation 5 times
    for i in range(5):
        # Step 5: Split the data into training and test sets
        (train_data, test_data) = df.randomSplit([0.8, 0.2], seed=i)

        # Step 6: Train the model
        cv_model = crossval.fit(train_data)

        # Step 7: Make predictions on the test data
        predictions = cv_model.transform(test_data)

        # Step 8: Evaluate the model based on what file we are using
        if file_name == "iris.csv":
            accuracy = evaluator.evaluate(predictions)
            avg_value += accuracy
        else:
            r2 = evaluator.evaluate(predictions)
            avg_value += r2

    avg_value /= 5

    spark.stop()

    end_time = time.time()

    total_time = end_time - start_time

    runtimes.append(total_time)
    accuracies.append(avg_value)


#this is to plot our executor cores/num cores to see how it affects runtime
#config_labels = []
#for config in configurations:
    #label = "Instances: " + config[0] + " Cores: " + config[1]
    #config_labels.append(label)

#this is to plot our threads and how it affects runtime
#config_labels = []
#for config in config_parallel:
    #label = "Threads : " + config
    #config_labels.append(label)

#this is to plot how repartitions affect our results
#config_labels = []
#for config in repartitions:
    #label = "Repartitions: " + str(config)
    #config_labels.append(label)

# Plot for runtimes
#plt.figure(figsize=(10, 5))
#plt.bar(config_labels, runtimes, color='skyblue')
#plt.xlabel('Configuration')
#plt.ylabel('Runtime')
#plt.title('Runtimes for Different Configurations')
#plt.xticks(rotation=45)
#plt.grid(axis='y')
#plt.tight_layout()
##plt.show()

# Plot for values
#plt.figure(figsize=(10, 5))
#plt.bar(config_labels, accuracies, color='lightgreen')
#plt.xlabel('Configuration')
#plt.ylabel('Value')
#plt.title('Values for Different Configurations')
#plt.xticks(rotation=45)
#plt.grid(axis='y')
##plt.tight_layout()
#plt.show()
