package org.ar.ml.examples.weka;

import org.junit.Test;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.io.File;
import java.nio.file.Paths;

import static org.junit.Assert.assertTrue;

/**
 * {@link SimpleLinearRegression} unit test.
 */
public class ComplexLinearRegressionTest {
  private static final String DATA_SET_PATH = "dou/2016_may_mini.csv";
  private static final String TARGET_VARIABLE_NAME = "Зарплата.в.месяц";
  private static final String EXPERIENCE_FEATURE_NAME = "exp";
  private static final String AGE_FEATURE_NAME = "Возраст";
  private static final String SALARY_CHANGE_FEATURE_NAME = "Изменение.зарплаты.за.12.месяцев";
  private static final String CURRENT_JOB_EXP_FEATURE_NAME = "current_job_exp";

  @Test
  public void predictSalaryForDifferentCityTest() throws Exception {
    //GIVEN
    File dataSetFile = Paths.get(getClass().getClassLoader().getResource(DATA_SET_PATH).toURI()).toFile();

    ComplexLinearRegression complexLinearRegression = new ComplexLinearRegression(dataSetFile,
        true, 2);

    //WHEN
    Instances preparedDataSet = complexLinearRegression.getDataSet();
    Instance searchInstance1 = buildTestInstance(preparedDataSet, preparedDataSet.firstInstance(),
        "Киев", "Senior_Software_Engineer", 7, 27);

    Instance searchInstance2 = buildTestInstance(preparedDataSet, preparedDataSet.lastInstance(),
        "Николаев", "Senior_Software_Engineer", 7, 27);

    //THEN
    double searchInstance1Result = complexLinearRegression.predict(searchInstance1);
    double searchInstance2Result = complexLinearRegression.predict(searchInstance2);

    assertTrue(searchInstance1Result > searchInstance2Result);
  }

  @Test
  public void predictSalaryForDifferentLevelsTest() throws Exception {
    //GIVEN
    File dataSetFile = Paths.get(getClass().getClassLoader().getResource(DATA_SET_PATH).toURI()).toFile();

    ComplexLinearRegression complexLinearRegression = new ComplexLinearRegression(dataSetFile,
        true, 2);

    //WHEN
    Instances preparedDataSet = complexLinearRegression.getDataSet();
    Instance searchInstance1 = buildTestInstance(preparedDataSet, preparedDataSet.firstInstance(),
        "Киев", "Software_Engineer", 7, 29);

    Instance searchInstance2 = buildTestInstance(preparedDataSet, preparedDataSet.lastInstance(),
        "Киев", "Senior_Software_Engineer", 9, 29);

    //THEN
    double searchInstance1Result = complexLinearRegression.predict(searchInstance1);
    double searchInstance2Result = complexLinearRegression.predict(searchInstance2);

    assertTrue(searchInstance1Result < searchInstance2Result);
  }

  private Instance buildTestInstance(Instances preparedDataSet, Instance searchInstance,
                                     String city, String title, int exp, int age) {
    setAllFeaturesToZero(searchInstance);
    searchInstance.setValue(getFeatureIndex(city, preparedDataSet), 1);
    searchInstance.setValue(getFeatureIndex(title, preparedDataSet), 1);
    searchInstance.setValue(getFeatureIndex(TARGET_VARIABLE_NAME, preparedDataSet), 0);
    searchInstance.setValue(getFeatureIndex(EXPERIENCE_FEATURE_NAME, preparedDataSet), exp);
    searchInstance.setValue(getFeatureIndex(AGE_FEATURE_NAME, preparedDataSet), age);
    searchInstance.setValue(getFeatureIndex(SALARY_CHANGE_FEATURE_NAME, preparedDataSet), 300);
    searchInstance.setValue(getFeatureIndex(CURRENT_JOB_EXP_FEATURE_NAME, preparedDataSet), 0.5);

    return searchInstance;
  }

  private void setAllFeaturesToZero(Instance instance) {
    for (int i = 0; i < instance.numValues(); i++) {
      instance.setValue(i, 0);
    }
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
