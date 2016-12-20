package org.ar.ml.examples.weka;

import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 *{@link ComplexLinearRegression} demonstrate how to use complex data set with textual features. It
 * replace each textual feature with set of new features that correspond to all possible original
 * feature values where all features that not correspond to real value set as 0 and the feature
 * that correspond to real feature values set as 1. Example :
 *
 * Input data :
 * No, City, Salary, Increase
 * 1, Kyiv, 2000, 300
 * 2, Lviv, 1900, 350
 *
 * Result :
 *
 * No, Kyiv, Lviv, Salary, Increate
 * 1, 1, 0, 2000, 300
 * 2, 0, 1, 1900, 350
 *
 */
public class ComplexLinearRegression {

  private final LinearRegression linearRegressionModel;
  private final Instances dataSet;

  public ComplexLinearRegression(File dataSetFile, boolean removeFirstColumn,
                                 int targetVariableIndex) throws Exception {

    dataSet = openTrainingDataSet(dataSetFile);
    String targetVariableName = dataSet.attribute(targetVariableIndex).name();

    normalizeDataSet(true);

    dataSet.setClassIndex(dataSet.attribute(targetVariableName).index());

    linearRegressionModel = new LinearRegression();
    //Disable selection methods to keep all features even it is not so important for result
    linearRegressionModel.setOptions(new String[]{"-S", "1"});
    linearRegressionModel.buildClassifier(dataSet);
  }

  /**
   * Predict target variable using trained linear regression model
   *
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
   * Normalize textual data and create suitable for linear regression data set.
   * @param removeFirstColumn Remove first column. Used if first column contain record ordering number.
   */
  private void normalizeDataSet(boolean removeFirstColumn) {
    if (removeFirstColumn) {
      dataSet.deleteAttributeAt(0);
    }

    Map<String, Set<String>> attributeToValuesMap = new HashMap<>();

    //Collect possible attributes and their unique values
    for (int i = 0; i < dataSet.numAttributes(); i++) {
      Attribute attribute = dataSet.attribute(i);
      if (!attribute.isNumeric()) {
        Set<String> values = new HashSet<>();
        for (int j = 0; j < attribute.numValues(); j++) {
          values.add(attribute.value(j).trim().replace(" ", "_"));
        }
        attributeToValuesMap.put(attribute.name(), values);
      }
    }

    //Populate data set with new features (each unique value) and then define value if this new
    // feature correspond to textual value in attribute that replacing
    attributeToValuesMap.forEach((s, strings) -> {
      for (String value : strings) {
        dataSet.insertAttributeAt(new Attribute(value), 0);
      }
      for (String value : strings) {
        dataSet.listIterator()
            .forEachRemaining(instance -> instance
                .setValue(dataSet.attribute(value).index(),
                    instance.stringValue(dataSet.attribute(s).index()).equals(value) ? 1 : 0));
      }
    });

    dataSet.deleteAttributeType(Attribute.NOMINAL);
    dataSet.deleteStringAttributes();

  }

  private Instances openTrainingDataSet(File dataSet) throws IOException {
    CSVLoader loader = new CSVLoader();
    loader.setSource(dataSet);
    return loader.getDataSet();
  }
}
