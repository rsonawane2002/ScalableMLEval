import time
import sys 
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


#these are the configurations for our executor_cores/ instances to see how our runtimes/efficiency changes
configurations = [["2", "2"], ["20", "4"], ["50", "6"], ["100", "8"]]
#these are the configurations for the number of threads we run on 
config_parallel = ["1", "5", "50", "100", "500"]

#these are the configurations for how we shuffle and repartition our dataframe
repartitions = [2, 5, 25, 50, 100]

runtimes = []
accuracie_graph = []

for i in range(1):

     #instances = config[0]
    #num_cores = config[1]

    start_time = time.time()

    # Step 1: Create a SparkSession
    spark = SparkSession.builder.appName("Multilayer Perceptron Classifier Example").getOrCreate()
    #spark = SparkSession.builder.appName("Multilayer Perceptron Classifier Example").config("spark.executor.instances", instances).config("spark.executor.cores", num_cores).getOrCreate()
    ##spark = SparkSession.builder.appName("Random Forest Example").config("spark.default.parallelism", config).getOrCreate()

    filename = "iris.csv"
    sepa = ','
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        sepa = ','

    # Step 2: Load the dataset into PySpark DataFrame
    data_df = spark.read.csv(filename, header=True, inferSchema=True, sep=',')

    #data_df = data_df.repartition(5)

    # Step 3: Prepare the features and label for model training
    if(filename == "iris.csv"):
        feature_cols = ["sepallength", "sepalwidth", "petallength", "petalwidth"]
        label = "variety"

    else:
        #get the features manually from the dataset
        feature_cols = ["Clump_Thickness", "Cell_Size_Uniformity", "Cell_Shape_Uniformity", "Marginal_Adhesion", "Single_Epi_Cell_Size", "Bare_Nuclei",	"Bland_Chromatin", "Normal_Nucleoli", "Mitoses"]
        label = "class"

        #transform the data to be able to be fed into the model using a vector assembler
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    data_df = assembler.transform(data_df)

    indexer = StringIndexer(inputCol=label, outputCol="label")
    data_df = indexer.fit(data_df).transform(data_df)

    train_df, valid_df = data_df.randomSplit([0.8, 0.2], seed=42)


    #define our network structure and then train our model
    layers = [len(feature_cols), 10, 8, 6, 4]  # Input layer size, two hidden layers, output layer size
    mlp_classifier = MultilayerPerceptronClassifier(maxIter=200, layers=layers, blockSize=128)

    # Initialize list to store accuracy values
    accuracies = []

    # Perform k-fold cross-validation
    for i in range(5):
        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
        param_grid = ParamGridBuilder().build()

        cv = CrossValidator(estimator=mlp_classifier, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=10, seed=42)
        cvModel = cv.fit(train_df)

        # Make predictions and evaluate on the validation set
        predictions = cvModel.transform(valid_df)
        accuracy = evaluator.evaluate(predictions)
        accuracies.append(accuracy)
        print("Validation accuracy: ", accuracy)

    # Calculate average accuracy across all runs
    average_accuracy = sum(accuracies) / len(accuracies)
    print("Average accuracy across 5 runs:", average_accuracy)
    accuracie_graph.append(average_accuracy)

    end_time = time.time()

    total_time = end_time - start_time

    print("Time elapsed", total_time)

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