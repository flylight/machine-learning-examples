# machine-learning-examples
Machine learning examples using popular approaches like Linear Regression, Logistic Regression (classification) and Neural Network on different solutions

**Training data set was took from public resource provided by [DOU.ua](https://dou.ua/) and does not
contain any sensitive information.**

## Linear regression example
- module : linear-regression

### Difference between Simple and Complex linear regression implementation

Both approaches retrieve data from CSV data set file provided by user but SimpleLinearRegression will removes all
textual features and keeps only numerical. It will simplify argolithm but decrease precision of prediction result. ComplexLinearRegression will process all data set and replace all textual features with a set of possible values as feature
where corresponded feature (value) will be set as 1 and all other as 0. This approach give more exact result based on all features in data set.

Example of data normalization in ComplexLinearRegression implementation :

*Inputed Data Set :*

|No        | City           | Salary  |  Increase  |
| -------- |:-------------:| :-----:|-----:|
| 1        | Kyiv          | $2000  | 300  |
| 2        | Lviv          | $2000  | 100  |
| 3        | Odessa        | $1900  | 300  |



*Normalized Data Set:*

|No        | Kyiv   |  Lviv  |  Odessa | Salary  |  Increase  |
| -------- |:------:|:------:|:-------:|:-------:|-----:|
| 1        | 1      |   0    |    0    | $2000  | 300  |
| 2        | 0      |   1    |    0    | $2000  | 100  |
| 3        | 0      |   0    |    1    | $1900  | 300  |

#### Weka implementation
- Package : package org.ar.ml.examples.weka;
- Class : SimpleLinearRegression.java or ComplexLinearRegression.java

This example show how to use [Weka](http://www.cs.waikato.ac.nz/ml/weka/) library to build simple
linear regression model using data set based only on Numerical features.



*Example for Simple Linear Regression usage*

```
// Create simple regression regression class using data set file, target variable index and define TRUE as we have to
// remove first column because it contains ordering numer of record (instance)
SimpleLinearRegression simpleLinearRegression = new SimpleLinearRegression(new File("path_to_csv_data_set"), 0, true);

// Using simple linear regression to predict target variable
double result = simpleLinearRegression.predict(myInsatnce);
```

## Logistic regression example
- module : logistic-regression

This kind of regression using to classify data and as result we expect to receive class identifier 
instead of real number.

#### Weka implementation
- Package : package org.ar.ml.examples.weka;
- Class : LogisticRegression.java

The main difference is that we have to use non numerical target variable to build classification 
algorithm. In example (test class "LogisticRegressionTest.java") I used "city" feature as target variable
and build model that can predict into which city I have to go work according to my salary expectation,
age, experience etc. 

This example id not good enough as we have very large training data set and to may different classes
(24 different cities in data set). So for me the training of such motel took about 24 hours. 

To make it faster I recommend to use Neural network or decrease set of features, for example it would be easier
for model to learn not more than 3-5 classes. It will be possible if we use salary indication and create 
tree classes like "Junior Engineer", "Middle Engineer", "Senior Engineer".

*Example for Logistic regression usage*

```
//Build logistic regression used data set file and define removing of ordering line and target varuable index
LogisticRegression complexLinearRegression = new LogisticRegression(dataSetFile, true, 1);

//Expecting to retrieve predicted result as class defined in datased as terget variable
String predictedResult = complexLinearRegression.predict(candidateInsatnce);

```
