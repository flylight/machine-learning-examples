package org.ar.ml.examples.weka;

import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;
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

  /**
   * Read CSV data set into {@link Instances} data container.
   *
   * @param dataSet Data set file in CSV format.
   * @return {@link Instances} data set container.
   * @throws IOException May throe exception when read data from file.
   */
  public Instances openTrainingDataSet(File dataSet) throws IOException {
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
  public Instances filterOutAllTextualFeatures(Instances rawDataSet, boolean removeFirstColumn) throws Exception {
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

  /**
   * Build {@link LinearRegression} model based on training data set and target variable index.
   *
   * @param trainingDataSet Training data set.
   * @param targetVariableIndex Target valuable index.
   * @return Trained and ready to use LinearRegression model.
   * @throws Exception May throw exception when build classifier.
   */
  public LinearRegression buildLinearRegressionModel(Instances trainingDataSet, int targetVariableIndex) throws Exception {
    trainingDataSet.setClassIndex(targetVariableIndex);

    LinearRegression linearRegressionModel = new LinearRegression();
    //Disable selection methods to keep all features even it is not so important for result
    linearRegressionModel.setOptions(new String[]{"-S", "1"});
    linearRegressionModel.buildClassifier(trainingDataSet);

    return linearRegressionModel;
  }
}
