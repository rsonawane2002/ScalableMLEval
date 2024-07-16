import time
import sys 
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#these are the configurations for our executor_cores/ instances to see how our runtimes/efficiency changes
configurations = [["2", "2"], ["20", "4"], ["50", "6"], ["100", "8"]]


#these are the configurations for the number of threads we run on 
config_parallel = ["1", "5", "50", "100", "500"]

#these are the configurations for how we shuffle and repartition our dataframe
repartitions = [2, 5, 25, 50, 100]

accuracies = []
runtimes = []

for i in range(1): 
    start_time = time.time()

     #instances = config[0]
    #num_cores = config[1]
    
    # Step 1: Create a SparkSession
    spark = SparkSession.builder .appName("Naive Bayes Example").config("spark.some.config.option", "some-value").getOrCreate()
    #spark = SparkSession.builder.appName("Naive Bayes Example").config("spark.default.parallelism", config).getOrCreate()
   # spark = SparkSession.builder.appName("Naive Bayes Example").config("spark.executor.instances", instances).config("spark.executor.cores", num_cores).getOrCreate()
    filename = "iris.csv"
    if(len(sys.argv) > 1):
        filename = sys.argv[1]



    # Step 2: Load the dataset into PySpark DataFrame
    data_df = spark.read.csv(filename, header=True, inferSchema=True, sep = ',')

    #data_df = data_df.repartition(config)

    # Step 3: Prepare the features for model training
    feature_cols = ["sepallength", "sepalwidth", "petallength", "petalwidth"]
    label = "variety"
    columns = []


    #large dataset, no need to k fold validate 
    if(filename == 'dota2Test.csv'):
        columns = ["win","clusterid", "gamemode", "gametype"]
        for i in range(1, 114):
            columns.append("hero" + str(i))
        
        feature_cols = columns[1:]
        data_df = data_df.toDF(*columns)
        
        label_col = data_df.columns[0]  # Assuming label column is the first column
        feature_cols = data_df.columns[1:]  # Assuming features start from the second column

        # Prepare the features for model training
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        data_df = assembler.transform(data_df)

        # Split data into 90% train and 10% test
        train_df, test_df = data_df.randomSplit([0.9, 0.1], seed=42)

        # Train a Naive Bayes model on the dataset
        naive_bayes = NaiveBayes(labelCol=label_col)
        model = naive_bayes.fit(train_df)

        # Make predictions on the test data
        predictions = model.transform(test_df)

        # Evaluate the model on precision and accuracy
        evaluator = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)

        precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
        print("Precision:", precision, "  Accuracy: ", accuracy)
        accuracies.append(accuracy)

    else:
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        data_df = assembler.transform(data_df)

        #index it to label
        indexer = StringIndexer(inputCol=label, outputCol="label")
        data_df = indexer.fit(data_df).transform(data_df)


        k = 10
        # Number of runs
        num_runs = 5
        # Initialize accuracy list
        accuracy_list = []

        # Perform cross-validation and average accuracy over 100 runs
        for a in range(num_runs):
            # Split data into k folds
            folds = data_df.randomSplit([1.0 / k] * k, seed=42)
            avg_accuracy = 0.0

            for i in range(k):
                # Prepare training and testing data for this fold
                test_df = folds[i]
                train_df = spark.createDataFrame(spark.sparkContext.emptyRDD(), data_df.schema)
                for j in range(k):
                    if j != i:
                        train_df = train_df.union(folds[j])

                # Train model
                naive_bayes = NaiveBayes()
                model = naive_bayes.fit(train_df)

                # Make predictions
                predictions = model.transform(test_df)

                # Evaluate accuracy
                evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
                accuracy = evaluator.evaluate(predictions)
                avg_accuracy += accuracy

            # Compute average accuracy for this run
            avg_accuracy /= k
            accuracy_list.append(avg_accuracy)

        # Calculate average accuracy over all runs
        average_accuracy = sum(accuracy_list) / len(accuracy_list)
        print("Average accuracy over 100 runs:", average_accuracy)
        accuracies.append(average_accuracy)


    spark.stop()

    end_time = time.time()
    total_time = end_time - start_time
    print("Total time taken:", total_time)
    runtimes.append(total_time)

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
#plt.show()

# Plot for values
#plt.figure(figsize=(10, 5))
#plt.bar(config_labels, accuracies, color='lightgreen')
#plt.xlabel('Configuration')
#plt.ylabel('Value')
#plt.title('Values for Different Configurations')
#plt.xticks(rotation=45)
#plt.grid(axis='y')
#plt.tight_layout()
#plt.show()
