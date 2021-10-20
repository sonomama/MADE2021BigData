Implementation of a simple linear regression model in scala using breeze library. 
The structure of the project is shown below:
<img width="284" alt="Screenshot 2021-10-20 at 17 23 09" src="https://user-images.githubusercontent.com/25647155/138132522-48f19f96-88dc-4854-9f7f-8dd8db1e042e.png">

I generated a piece of data for this task, to make sure that the code works correctly. 
The data is stored in src/main/data_example. 

The code expects the input_file_path and output_file_path for the metrics. 
I use the data_example directory for this purposes and the metrics for the dummy_data.csv is also stored there.
In the main I load the data, do the crossvalidation and then do the train-test split and validation on the test data.
Simple logging is also implemented.
