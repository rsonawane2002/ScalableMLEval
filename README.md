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


There is a video documenting my findings in the files. Here are charts below describing the relationships I found:

<img width="709" alt="Screen Shot 2024-07-17 at 4 09 29 PM" src="https://github.com/user-attachments/assets/694ec3e7-f00c-4ab3-b574-c8af96ce3d92">

<img width="731" alt="Screen Shot 2024-07-17 at 4 08 59 PM" src="https://github.com/user-attachments/assets/0946c817-f785-4085-a98f-1586313e8525">

<img width="751" alt="Screen Shot 2024-07-17 at 4 13 25 PM" src="https://github.com/user-attachments/assets/f99d97aa-f050-4d6c-98ef-743fe97207ac">


<img width="752" alt="Screen Shot 2024-07-17 at 4 14 59 PM" src="https://github.com/user-attachments/assets/33bfe42f-52d9-4b2a-b761-6a95b214986d">

<img width="728" alt="Screen Shot 2024-07-17 at 4 13 45 PM" src="https://github.com/user-attachments/assets/5bfecac3-67b8-4316-9cef-9c50b819d3a6">

<img width="732" alt="Screen Shot 2024-07-17 at 4 12 28 PM" src="https://github.com/user-attachments/assets/c9e34f48-a310-4b6c-bd4f-6e7d8cd4186f">


<img width="744" alt="Screen Shot 2024-07-17 at 4 14 05 PM" src="https://github.com/user-attachments/assets/d6231039-109c-498c-8828-e4c12fd40fee">

<img width="736" alt="Screen Shot 2024-07-17 at 4 14 13 PM" src="https://github.com/user-attachments/assets/d1a475b6-9b6e-4075-bfde-9a7b51502aaf">

<img width="718" alt="Screen Shot 2024-07-17 at 4 09 55 PM" src="https://github.com/user-attachments/assets/b6969582-0c66-4b01-a7d3-d340c3021747">







