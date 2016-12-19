package org.ar.ml.examples.weka;

import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils;
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

  public Instances openTrainingDataSet(File dataSet) throws IOException {
    CSVLoader loader = new CSVLoader();
    loader.setSource(dataSet);
    return loader.getDataSet();
  }

  public Instances filterOutAllTextualFeatures(Instances rawDataSet) throws Exception {
    List<String> indexesToRemove = new ArrayList<>();
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

  public LinearRegression buildLinearRegressionModel(Instances trainingDataSet, int targetVariableIndex) throws Exception {
    trainingDataSet.setClassIndex(targetVariableIndex);

    LinearRegression linearRegressionModel = new LinearRegression();
    linearRegressionModel.buildClassifier(trainingDataSet);
    //just to show more information when print this object
    linearRegressionModel.setOutputAdditionalStats(true);

    return linearRegressionModel;
  }
}
