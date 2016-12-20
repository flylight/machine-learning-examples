package org.ar.ml.examples.weka;

import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

import java.io.File;
import java.io.IOException;

/**
 * This class show how to use Weka library to create simple linear regression model to predict
 * result using only number fields.
 */
public class SimpleLinearRegression {

  private final LinearRegression linearRegressionModel;
  private final Instances dataSet;
  /**
   * Create {@link SimpleLinearRegression} object. This object use provided data set file in CSV
   * format and other properties to build and train linear regression using just numerical features.
   * @param dataSetFile CSV data set file.
   * @param removeFirstColumn Remove first column or not identifier. Used ad TRUE when first column
   *                          represent order number for record in data set file.
   * @param targetVariableIndex Target variable index in data set.
   * @throws Exception May throw exception when initialising linear regression model.
   */
  public SimpleLinearRegression(File dataSetFile, boolean removeFirstColumn,
                                int targetVariableIndex) throws Exception {
    dataSet = openTrainingDataSet(dataSetFile);
    String targetVariableName = dataSet.attribute(targetVariableIndex).name();

    filterOutAllTextualFeatures(removeFirstColumn);

    dataSet.setClassIndex(dataSet.attribute(targetVariableName).index());

    linearRegressionModel = new LinearRegression();
    //Disable selection methods to keep all features even it is not so important for result
    linearRegressionModel.setOptions(new String[]{"-S", "1"});
    linearRegressionModel.buildClassifier(dataSet);
  }

  /**
   * Predict target variable using trained linear regression model
   * @param instanceToPredict Instance of data to be predicted.
   * @return predicted target variable.
   * @throws Exception May be thrown when predicting target variable.
   */
  public double predict(Instance instanceToPredict) throws Exception {
    return linearRegressionModel.classifyInstance(instanceToPredict);
  }

  public Instances getDataSet() {
    return dataSet;
  }

  /**
   * Read CSV data set into {@link Instances} data container.
   *
   * @param dataSet Data set file in CSV format.
   * @return {@link Instances} data set container.
   * @throws IOException May throe exception when read data from file.
   */
  private Instances openTrainingDataSet(File dataSet) throws IOException {
    CSVLoader loader = new CSVLoader();
    loader.setSource(dataSet);
    return loader.getDataSet();
  }

  /**
   * Delete all textual paramater and leave only numerical types to simplify processing.
   * @param removeFirstColumn Remove first column from data set or no. Used when first column
   *                          represent numerical order ot items in data set.
   */
  private void filterOutAllTextualFeatures(boolean removeFirstColumn) {
    if (removeFirstColumn) {
      dataSet.deleteAttributeAt(0);
    }
    dataSet.deleteAttributeType(Attribute.NOMINAL);
    dataSet.deleteStringAttributes();
  }
}
