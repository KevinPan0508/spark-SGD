import sys
import os
SPARK_HOME = "/opt/bitnami/spark"
os.environ["SPARK_HOME"] = SPARK_HOME
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
sys.path.append( SPARK_HOME + "/python")
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
import numpy as np
from pyspark import SparkConf, SparkContext
def getSparkContext():
     conf = (SparkConf().setMaster("local").setAppName("Logistic Regression").set("spark.executor.memory", "1g"))
     sc1 = SparkContext(conf = conf)
     return sc1

sc = getSparkContext()
data = sc.textFile("/home/data_banknote_authentication.txt")

def mapper(line):
    feats = line.strip().split(",")
    label = feats[len(feats) - 1]
    feats = feats[: len(feats) - 1]
    feats.insert(0,label)
    features = [ float(feature) for feature in feats ]
    return LabeledPoint(label, features)



parsedData = data.map(mapper)
model = LogisticRegressionWithSGD.train(parsedData)

labelsAndPreds = parsedData.map(lambda point: (int(point.label),model.predict(point.features.take(range(point.features.size)))))


trainErr = labelsAndPreds.filter(lambda v: v[0] != v[1]).count() / float(parsedData.count())

print("Training Error = " + str(trainErr))
