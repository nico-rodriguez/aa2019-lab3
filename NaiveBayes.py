"""
TODO
This module implement the Naive Bayes algorithm. It assumes that the instances are separated by class and that
the distributions of each attribute (normal distribution is assumed) are already calculated.
The classification function of this module returns a class number.
"""

import Evaluator
import NBParser
import Utils

# Implement the Naive Bayes algorithm.
# - 'attribute_distributions_per_class' is a dict with the distributions for each subclass
# - 'classes_distributions' is the percentage of that instances with that class in the dataset
# - 'instance' is the instance to classify
# - 'classes_labels' is a list with all class labels.
# Return the class number that classifies x with the K-NN algorithm.
def naive_bayes_classify_instance(classes_distributions, attribute_distributions_per_class, classes_labels, instance):
    max_index = 0
    max_probability = -1
    for class_index in range(len(classes_labels)):
        probability = classes_distributions[classes_labels[class_index]]
        distributions = attribute_distributions_per_class[class_index]
        for index in range(len(instance)-1): # -1 to not use the label
            # Dictionary: attribute number -> (distribution type, distribution parameters)
            (distr_type, distr_parameters) = distributions[index]
            attr_value = instance[index]
            if distr_type == "normal": # If distribution type == 'normal', distribution parameters == {'mean': m, 'variance': v}
                probability *= Utils.gaussian(distr_parameters["mean"], distr_parameters["variance"], attr_value)
            elif distr_type == "uniform": # If distribution type == 'uniform', distribution parameters == {v1: f1, ... , vn: fn}, where v1, ... , vn
            # are all the possible attribute values and f1, ... , fn are the frequencies of those values in the set
                probability *= distr_parameters[attr_value]
            else:
                raise Exception("Distribution type error")
        if probability > max_probability:
            max_probability = probability
            max_index = class_index
    # pass from label to index
    return max_index


# Classify using Naive Bayes
# - 'attribute_distributions_per_class' is a dict with the distributions for each subclass
# - 'classes_distributions' is the percentage of that instances with that class in the dataset
# - 'dataset' is the dataset to classify
# - 'classes_labels' is a list with all class labels.
# Return the classified elements as a list of tuples (label, guess)
def naive_bayes_classify_dataset(classes_distributions, attribute_distributions_per_class, classes_labels, dataset):
    result = []
    for instance in dataset:
        guess = naive_bayes_classify_instance(classes_distributions, attribute_distributions_per_class, classes_labels, instance)
        label = instance[-1]
        result.append([classes_labels[label], classes_labels[guess]])
    return result

if __name__ == '__main__':
    #iris
    distributions_dictionary = NBParser.naive_bayes_load_distributions(NBParser.naive_bayes_iris_distributions_file_name)
    validation_set = NBParser.naive_bayes_load_validation_instances(NBParser.naive_bayes_iris_validation_instances_file_name)
    classes_labels = ["Iris Setosa", "Iris Versicolour", "Iris Virginica"]
    classes_distributions = {"Iris Setosa" : 1/3, "Iris Versicolour" : 1/3, "Iris Virginica" : 1/3}
    classified_data = naive_bayes_classify_dataset(classes_distributions, distributions_dictionary, classes_labels, validation_set)
    Evaluator.evaluate_classifier(classified_data, classes_labels, 'naive_bayes_exp/iris.data')
    #covtype
    distributions_dictionary = NBParser.naive_bayes_load_distributions(NBParser.naive_bayes_covtype_distributions_file_name)
    validation_set = NBParser.naive_bayes_load_validation_instances(NBParser.naive_bayes_covtype_validation_instances_file_name)
    classes_labels = ["Spruce/Fir", "Lodgepole Pine", "Ponderosa Pine", "Cottonwood/Willow", "Aspen", "Douglas-fir", "Krummholz"]
    classes_distributions = {"Spruce/Fir" : 211840/581012, "Lodgepole Pine" : 283301/581012,
                             "Ponderosa Pine" : 35754/581012, "Cottonwood/Willow" : 2747/581012,
                            "Aspen" : 9493/581012, "Douglas-fir" : 17367/581012, "Krummholz" : 20510/581012}
    classified_data = naive_bayes_classify_dataset(classes_distributions, distributions_dictionary, classes_labels, validation_set)
    Evaluator.evaluate_classifier(classified_data, classes_labels, 'naive_bayes_exp/covtype.data')


