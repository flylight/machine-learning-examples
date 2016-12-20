# machine-learning-examples
Machine learning examples using popular approaches like Linear Regression, Logistic Regression (classification) and Neural Network on different solutions

## Linear regression example

### Difference between Simple and Complex linear regression implementation

Both approaches retrieve data from CSV data set file provided by user but SimpleLinearRegression will removes all
textual features and keeps only numerical. It will simplify argolithm but decrease precision of prediction result. ComplexLinearRegression will process all data set and replace all textual features with a set of possible values as feature
where corresponded feature (value) will be set as 1 and all other as 0. This approach give more exact result based on all features in data set.

#### Weka implementation
- Package : package org.ar.ml.examples.weka;
- Class : SimpleLinearRegression.java or ComplexLinearRegression.java

This example show how to use [Weka](http://www.cs.waikato.ac.nz/ml/weka/) library to build simple
linear regression model using data set based only on Numerical features.



**Example for Simple Linear Regression usage**

```
// Create simple regression regression class using data set file, target variable index and define TRUE as we have to
// remove first column because it contains ordering numer of record (instance)
SimpleLinearRegression simpleLinearRegression = new SimpleLinearRegression(new File("path_to_csv_data_set"), 0, true);

// Using simple linear regression to predict target variable
double result = simpleLinearRegression.predict(myInsatnce);
```


