# ScalableMLEval
Developed a scalable machine learning model evaluation system in PySpark, parallelizing computations across distributed nodes and comparing performance of multiple models on large datasets based on accuracy, precision, and resource utilization metrics




To run, simply do: python3 file_name.py


1) For naive_bayes, you can add parameter "dota2Test.csv" for the complex dataset
2) For random_forest, you can add parameter "gait.csv" for the complex dataset
3) For neural_net you can add parameter "cancer.csv" for the complex dataset


Note, I have commented out all the code that generates the plot and also uses PySpark's
resource allocation/parallelism to save runtime. 

To see/generate plots, run the given file on the iris.csv dataset (no additional paremeter after the filename)

Then, at the top of each file, you will see a configuration and a list. Each list has a description of what
resource it is testing.

Then, you need to modify the for loop to run over that list and then uncomment the code it is referencing.

If it is executor cores, comment out the sparksessionbuilder at the top that corresopnds to it, for example.

Then, go to the bottom and comment out the for loop corresponding to the resource you picked as well as the 
entire pyplot code. This should run the code for every given config on iris.csv and yield a graph of runtime/accuracy for the given config.

Please reach out to me if you have any trouble
