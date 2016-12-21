package org.ar.ml.examples.weka;

import weka.classifiers.functions.Logistic;
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
 *
 */
public class LogisticRegression {

  private final Logistic logisticRegressionModel;
  private final Instances dataSet;

  private int targetVariableIndex;


  public LogisticRegression(File dataSetFile, boolean removeFirstColumn,
                            int targetVariableIndex) throws Exception {
    this.targetVariableIndex = targetVariableIndex;

    dataSet = openTrainingDataSet(dataSetFile);
    normalizeDataSet(removeFirstColumn);

    dataSet.setClassIndex(this.targetVariableIndex);

    logisticRegressionModel = new Logistic();
    logisticRegressionModel.buildClassifier(dataSet);
  }

  public String predict(Instance instanceToPredict) throws Exception {
    return dataSet.classAttribute().value(
        (int)logisticRegressionModel.classifyInstance(instanceToPredict));
  }

  public Instances getDataSet() {
    return dataSet;
  }


  private void normalizeDataSet(boolean removeFirstColumn) {
    if (removeFirstColumn) {
      dataSet.deleteAttributeAt(0);
      targetVariableIndex--;
    }

    Map<String, Set<String>> attributeToValuesMap = new HashMap<>();

    //Collect possible attributes and their unique values
    for (int i = 0; i < dataSet.numAttributes(); i++) {
      Attribute attribute = dataSet.attribute(i);
      //Skip all numerical features and target variable
      if (!attribute.isNumeric() && attribute.index() != targetVariableIndex) {
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
        dataSet.insertAttributeAt(new Attribute(value), dataSet.numAttributes());
      }
      for (String value : strings) {
        dataSet.listIterator()
            .forEachRemaining(instance -> instance
                .setValue(dataSet.attribute(value).index(),
                    instance.stringValue(dataSet.attribute(s).index()).equals(value) ? 1 : 0));
      }
    });

    for (int i = 0; i < dataSet.numAttributes(); ) {
      Attribute attribute = dataSet.attribute(i);
      if (!attribute.isNumeric() && i != targetVariableIndex) {
        dataSet.deleteAttributeAt(i);
      } else {
        i++;
      }
    }
  }

  private Instances openTrainingDataSet(File dataSet) throws IOException {
    CSVLoader loader = new CSVLoader();
    loader.setSource(dataSet);
    return loader.getDataSet();
  }

}
