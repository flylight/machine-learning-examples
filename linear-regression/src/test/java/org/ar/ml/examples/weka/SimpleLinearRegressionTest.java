package org.ar.ml.examples.weka;

import org.junit.Test;

import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.nio.file.Paths;

import static org.junit.Assert.assertTrue;

/**
 * {@link SimpleLinearRegression} unit test.
 */
public class SimpleLinearRegressionTest {
  private static final String DATA_SET_PATH = "dou/2016_may_mini.csv";
  private static final String TARGET_VARIABLE_NAME = "Зарплата.в.месяц";
  private static final String EXPERIENCE_FEATURE_NAME = "exp";
  private static final String AGE_FEATURE_NAME = "Возраст";
  private static final String SALARY_CHANGE_FEATURE_NAME = "Изменение.зарплаты.за.12.месяцев";
  private static final String CURRENT_JOB_EXP_FEATURE_NAME = "current_job_exp";

  @Test
  public void predictSalaryTest() throws Exception {
    //GIVEN
    SimpleLinearRegression simpleLinearRegression = new SimpleLinearRegression();
    Instances rawDataSet = simpleLinearRegression
        .openTrainingDataSet(Paths.get(getClass().getClassLoader().getResource(DATA_SET_PATH).toURI()).toFile());

    Instances preparedDataSet = simpleLinearRegression.filterOutAllTextualFeatures(rawDataSet);

    LinearRegression linearRegression = simpleLinearRegression
        .buildLinearRegressionModel(preparedDataSet, getFeatureIndex(TARGET_VARIABLE_NAME, preparedDataSet));

    //WHEN
    Instance searchInstance1 = preparedDataSet.firstInstance();
    searchInstance1.setValue(getFeatureIndex(TARGET_VARIABLE_NAME, preparedDataSet), 0);
    searchInstance1.setValue(getFeatureIndex(EXPERIENCE_FEATURE_NAME, preparedDataSet), 7);
    searchInstance1.setValue(getFeatureIndex(AGE_FEATURE_NAME, preparedDataSet), 27);
    searchInstance1.setValue(getFeatureIndex(SALARY_CHANGE_FEATURE_NAME, preparedDataSet), 300);
    searchInstance1.setValue(getFeatureIndex(CURRENT_JOB_EXP_FEATURE_NAME, preparedDataSet), 0.5);

    Instance searchInstance2 = preparedDataSet.lastInstance();
    searchInstance2.setValue(getFeatureIndex(TARGET_VARIABLE_NAME, preparedDataSet), 0);
    searchInstance2.setValue(getFeatureIndex(EXPERIENCE_FEATURE_NAME, preparedDataSet), 7);
    searchInstance2.setValue(getFeatureIndex(AGE_FEATURE_NAME, preparedDataSet), 31);
    searchInstance2.setValue(getFeatureIndex(SALARY_CHANGE_FEATURE_NAME, preparedDataSet), 100);
    searchInstance2.setValue(getFeatureIndex(CURRENT_JOB_EXP_FEATURE_NAME, preparedDataSet), 0.5);

    //THEN
    double searchInstance1Result = linearRegression.classifyInstance(searchInstance1);
    double searchInstance2Result = linearRegression.classifyInstance(searchInstance2);

    assertTrue(searchInstance1Result > searchInstance2Result);
  }

  private int getFeatureIndex(String featureName, Instances dataSet) {
    for (int i = 0; i < dataSet.numAttributes(); i++) {
      Attribute attribute = dataSet.attribute(i);
      if (attribute.name().endsWith(featureName)) {
        return i;
      }
    }
    throw new IllegalArgumentException("Target variable" + TARGET_VARIABLE_NAME + " not found.");
  }
}
