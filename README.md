# machine-learning-examples
Machine learning examples using popular approaches like Linear Regression, Logistic Regression (classification) and Neural Network on different solutions

## Simple linear regression example

### Weka implementation
- Package : package org.ar.ml.examples.weka;
- Class : SimpleLinearRegression

This example show how to use [Weka](http://www.cs.waikato.ac.nz/ml/weka/) library to build simple
linear regression model using data set based only on Numerical features.

You can use any CSV data set but before build Linear Regression object you have to use 
filtering to remove all textual features. 

**Example**

```
// Create simple regression provider 
SimpleLinearRegression simpleLinearRegression = new SimpleLinearRegression();

// Use provider to read data set from CSV file
Instances rawDataSet = simpleLinearRegression.openTrainingDataSet(new File("path_to_csv_data_set"));

// Filterout all textual features. Also you can filterount first column in dataset if it contain ordering number
Instances preparedDataSet = simpleLinearRegression.filterOutAllTextualFeatures(rawDataSet, true);

//Build Linear Regression from prepared data set and specify targed variable index
LinearRegression linearRegression = simpleLinearRegression.buildLinearRegressionModel(preparedDataSet, 0);

```

Then you can create your own instance to predict target variable

**Example**

`double result = linearRegression.classifyInstance(myInsatnce);`
