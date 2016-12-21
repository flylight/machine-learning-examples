package org.ar.ml.examples.weka;

import org.junit.Test;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.io.File;
import java.nio.file.Paths;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 */
public class LogisticRegressionTest {
  private static final String DATA_SET_PATH = "dou/2016_may_mini.csv";

  @Test
  public void predictSalaryForDifferentCityTest() throws Exception {
    //GIVEN
    File dataSetFile = Paths.get(getClass().getClassLoader().getResource(DATA_SET_PATH).toURI()).toFile();

    LogisticRegression complexLinearRegression = new LogisticRegression(dataSetFile,
        true, 1);

    Instance candidateInsatnce = complexLinearRegression.getDataSet().firstInstance();

    System.out.println("ANSWER : " + complexLinearRegression.predict(candidateInsatnce));

    //assertEquals(candidateInsatnce.attribute(1).value(0), predictedCity);
  }
}
