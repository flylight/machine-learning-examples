package org.ar.ml.examples.weka;

import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * This class show how to use Weka library to create simple linear regression model to predict
 * result using only number fields.
 */
public class SimpleLinearRegression {

  private final LinearRegression linearRegressionModel;
  private final Instances trainingDataSet;
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
    Instances rawDataSet = openTrainingDataSet(dataSetFile);
    String targetVariableName = rawDataSet.attribute(targetVariableIndex).name();

    trainingDataSet = filterOutAllTextualFeatures(rawDataSet, removeFirstColumn);

    trainingDataSet.setClassIndex(trainingDataSet.attribute(targetVariableName).index());

    linearRegressionModel = new LinearRegression();
    //Disable selection methods to keep all features even it is not so important for result
    linearRegressionModel.setOptions(new String[]{"-S", "1"});
    linearRegressionModel.buildClassifier(trainingDataSet);
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

  public Instances getTrainingDataSet() {
    return trainingDataSet;
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
   * Filter out all textual features as this simple version can works only with numerical feature types.
   * @param rawDataSet Raw data set.
   * @param removeFirstColumn Remove first column if it contain ordering number.
   * @return Filtered data set.
   * @throws Exception may throw exception when process data set.
   */
  private Instances filterOutAllTextualFeatures(Instances rawDataSet, boolean removeFirstColumn) throws Exception {
    List<String> indexesToRemove = new ArrayList<>();
    if (removeFirstColumn) {
      //Add first column into removing as it show order number in data set
      indexesToRemove.add("1");
    }

    for (int i = 0; i < rawDataSet.numAttributes(); i++) {
      Attribute attribute = rawDataSet.attribute(i);
      if (!attribute.isNumeric()) {
        //Weka contain attributes by indexes started from 0 (zero) but removing functionality
        //use indexes starting from 1 so index should be incremented by 1.
        indexesToRemove.add(String.valueOf(i+1));
      }
    }

    String[] options = new String[2];
    options[0] = "-R";
    options[1] = String.join(",", indexesToRemove);

    Remove remove = new Remove();
    remove.setOptions(options);
    remove.setInputFormat(rawDataSet);

    return Filter.useFilter(rawDataSet, remove);
  }
}
