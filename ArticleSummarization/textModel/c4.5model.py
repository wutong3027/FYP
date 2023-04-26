from py4j.java_gateway import JavaGateway, GatewayParameters
from weka.core.converters import Loader
from weka.classifiers import Classifier, Evaluation
from weka.classifiers.trees import J48
from weka.filters import Filter
import pandas as pd

def main():
    gateway = JavaGateway(gateway_parameters=GatewayParameters(auto_convert=True))
    jvm = gateway.jvm

    # Load the dataset from CSV file
    df = pd.read_csv("scientific_paper.csv")

    # Convert the dataframe to a Weka dataset
    loader = Loader(classname="weka.core.converters.CSVLoader")
    data = loader.load_file("scientific_paper.csv")
    data.class_is_last()

    # Apply a filter to preprocess the data
    filter = Filter(classname="weka.filters.unsupervised.attribute.StringToWordVector")
    filter.inputformat(data)
    data = filter.filter(data)

    # Split the dataset into training and testing sets
    train_data, test_data = data.train_test_split(jvm.java.lang.Integer(66))

    # Train the C4.5 classifier on the training set
    cls = Classifier(classname="weka.classifiers.trees.J48")
    cls.build_classifier(train_data)

    # Evaluate the classifier on the testing set
    eval = Evaluation(train_data)
    eval.test_model(cls, test_data)

    # Print the accuracy of the classifier
    print("Accuracy:", eval.percent_correct)

if __name__ == "__main__":
    main()
