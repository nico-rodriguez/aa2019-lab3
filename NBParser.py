"""
Read the data sets Iris and Cover type and process the instances, so that it can be used by the algorithms.

For both Naive Bayes and K-NN:
+ convert Iris class from str to int. The mapping is Iris-setosa <-> 0, Iris-versicolor <-> 1 and Iris-virginica <-> 2.

For Naive Bayes:
+ convert binary attributes of Cover type to integers.
+ for categorical attributes, take into account the necessity of using an m-estimator.
+ calculate and store the distributions of each attribute (assuming normal distribution).
+ return instances separated by class.
"""

import ast
import json
import random
import statistics
import Utils

"""
Parsers for Naive Bayes.
PRE CONDITION: the iris data set file was already parsed with __parse_iris_class_label.
"""


naive_bayes_directory = 'naive_bayes/'
naive_bayes_iris_distributions_file_name = naive_bayes_directory + 'iris_distributions.json'
naive_bayes_iris_instances_file_name = naive_bayes_directory + 'iris_instances.json'
naive_bayes_iris_validation_instances_file_name = naive_bayes_directory + 'iris_validation.data'
naive_bayes_covtype_distributions_file_name = naive_bayes_directory + 'covtype_distributions.json'
naive_bayes_covtype_instances_file_name = naive_bayes_directory + 'covtype_instances.json'
naive_bayes_covtype_validation_instances_file_name = naive_bayes_directory + 'covtype_validation.data'


# Parse the cover type instances, changing the binary attributes for numerical values in cover type data set case.
# Split the instances in validation set and training set.
# Create a dictionary: class label -> list of instances of that class, for the training set and the validation set.
# Save the dictionaries in naive_bayes_instances (json) and return them.
def __naive_bayes_parse_instances(data_set_file_path, training_proportion):
    with open(data_set_file_path, 'r') as data_set_file:
        instances_lines = data_set_file.readlines()
    # Parse each instance line to a list (without the class label at the end) and create a
    # dictionary: class label -> list of instances of that class.
    instances_per_class = {}
    if 'iris' in data_set_file_path:
        classes_number = 3
    elif 'covtype' in data_set_file_path:
        classes_number = 7
    else:
        raise Exception('Parser.naive_bayes_parse_binary_attributes: "iris" or "covtype" not present in data set file' +
                        ' path')
    instances_list_with_class = []
    # Initialize empty dictionary
    for c in range(classes_number):
        instances_per_class[c] = []
    # Split each instance according to it's class
    for instance_line in instances_lines:
        instance_tokens = instance_line.split(',')
        class_label = Utils.num(instance_tokens[-1].rstrip())
        # Adjust the class label range of cover type from [1,...,7] to [0,...,6]
        class_label = class_label if 'iris' in data_set_file_path else class_label - 1
        if 'iris' in data_set_file_path:
            # Only add attribute values to the instance list
            instance = [Utils.num(instance_tokens[i]) for i in range(len(instance_tokens)-1)]
        else:
            # Only add numeric attribute values to the instance list
            instance = [Utils.num(instance_tokens[i]) for i in range(10)]
            # Convert the first four binary attributes to an integer
            instance.append(instance_tokens[10:14].index('1'))
            # Convert the last forty binary attributes to an integer
            instance.append(instance_tokens[14:54].index('1'))
        instances_list_with_class.append((instance, class_label))

    # Split the instances in training set and validation set
    random.shuffle(instances_list_with_class)
    validation_instances_number = int(len(instances_list_with_class) - len(instances_list_with_class) *
                                      training_proportion)
    validation_instances_list_with_class = instances_list_with_class[0:validation_instances_number]
    instances_list_with_class = instances_list_with_class[validation_instances_number:]
    validation_instances_with_classes = []
    for validation_instance, class_label in validation_instances_list_with_class:
        # Add to validation instances list with class label
        validation_instance.append(class_label)
        validation_instances_with_classes.append(validation_instance)

    for instance, class_label in instances_list_with_class:
        instances_per_class[class_label].append(instance)

    # Save the instances dictionary as a json file
    if 'iris' in data_set_file_path:
        instances_file_name = naive_bayes_iris_instances_file_name
    else:
        instances_file_name = naive_bayes_covtype_instances_file_name
    with open(instances_file_name, 'w') as naive_bayes_instances_file:
        json.dump(instances_per_class, naive_bayes_instances_file)

    # Save the validation instances dictionary as a json file
    if 'iris' in data_set_file_path:
        validation_instances_file_name = naive_bayes_iris_validation_instances_file_name
    else:
        validation_instances_file_name = naive_bayes_covtype_validation_instances_file_name
    with open(validation_instances_file_name, 'w') as naive_bayes_validation_instances_file:
        for validation_instance in validation_instances_with_classes:
            naive_bayes_validation_instances_file.write(str(validation_instance) + '\n')

    return instances_per_class


# Main parser function.
# Generate the files with the instances divided by class and the file with the pre computed distributions.
def __naive_bayes_parser(data_set_file_path, training_proportion):
    instances_per_class = __naive_bayes_parse_instances(data_set_file_path, training_proportion)

    if 'iris' in data_set_file_path:
        attributes_number = 4
        classes_number = 3
    else:
        attributes_number = 12
        classes_number = 7

    # Dictionary: class number -> distribution dictionary
    distribution_per_class = {c: {} for c in range(classes_number)}

    for c in range(classes_number):
        # Dictionary: attribute number -> (distribution type, distribution parameters)
        # If distribution type == 'normal', distribution parameters == {'mean': m, 'variance': v}
        # If distribution type == 'uniform', distribution parameters == {v1: f1, ... , vn: fn}, where v1, ... , vn
        # are all the possible attribute values and f1, ... , fn are the frequencies of those values in the set of
        # instances with the same class.
        distribution = {}

        for attribute in range(attributes_number):
            distribution_type = 'uniform' if Utils.is_categorical(attribute) else 'normal'
            attribute_values_list = [instance[attribute] for instance in instances_per_class[c]]
            if distribution_type == 'uniform':
                # Dictionary: attribute value -> frequency of value
                frequency = {}
                for attribute_value in Utils.categorical_attribute_values(attribute):
                    attribute_value_frequency = attribute_values_list.count(attribute_value)/len(attribute_values_list)
                    # Check if we need to use an m-estimator
                    if attribute_value_frequency == 0:
                        equivalent_sample_size = len(attribute_values_list)
                        probability_estimate = 1/Utils.categorical_attribute_values_number(attribute)
                        frequency[attribute_value] = equivalent_sample_size*probability_estimate/(
                            equivalent_sample_size + len(attribute_values_list)
                        )
                    # If we don't need an m-estimator, just assign the frequency of the value as the probability.
                    else:
                        frequency[attribute_value] = attribute_value_frequency
                distribution[attribute] = (distribution_type, frequency)
            else:
                mean = statistics.mean(attribute_values_list)
                variance = statistics.variance([instance[attribute] for instance in instances_per_class[c]], mean)
                distribution[attribute] = (distribution_type, {'mean': mean, 'variance': variance})

        distribution_per_class[c] = distribution

    # Save the distributions to a json file
    if 'iris' in data_set_file_path:
        distributions_file_name = naive_bayes_iris_distributions_file_name
    else:
        distributions_file_name = naive_bayes_covtype_distributions_file_name
    with open(distributions_file_name, 'w') as distributions_file:
        json.dump(distribution_per_class, distributions_file)


# Functions for loading the instances and distributions files

# Loads the dictionary of distributions. Returns the dictionary.
def naive_bayes_load_distributions(dictionary_file_path):
    with open(dictionary_file_path, 'r') as distributions_file:
        # With json.load, the int keys are loaded as string keys and the tuples are loaded as lists,
        # because json.dump saves the keys as strings and the tuples as list.
        dictionary_string_keys = json.load(distributions_file)
        # Convert the string keys to int keys, if possible, and the list to tuples
        dictionary_int_keys = {int(c): {} for c in dictionary_string_keys}
        for c in dictionary_string_keys:   # c is a str
            for attribute in dictionary_string_keys[c]:    # attribute is a str
                if dictionary_string_keys[c][attribute][0] == 'normal':
                    dictionary_int_keys[int(c)][int(attribute)] = ('normal',
                                                                   dictionary_string_keys[c][attribute][1].copy())
                elif dictionary_string_keys[c][attribute][0] == 'uniform':
                    frequency = {}
                    # Convert attribute value key from str to int
                    for attribute_value in dictionary_string_keys[c][attribute][1]:
                        frequency[int(attribute_value)] = dictionary_string_keys[c][attribute][1][attribute_value]
                    dictionary_int_keys[int(c)][int(attribute)] = ('uniform', frequency)
    return dictionary_int_keys


# Loads the dictionary of instances per class. Returns the dictionary.
def naive_bayes_load_instances(instances_file_path):
    with open(instances_file_path, 'r') as instances_file:
        # With json.load, the int keys are loaded as string keys, because json.dump saves the keys as strings.
        instances_per_class_string_keys = json.load(instances_file)
    # Convert the string keys to int keys
    instances_per_class = {}
    for class_string in instances_per_class_string_keys:
        instances_per_class[int(class_string)] = instances_per_class_string_keys[class_string]
    return instances_per_class


# Loads the instances for validation. The instances are lists of attributes with the class label at the end.
def naive_bayes_load_validation_instances(validation_instances_file_path):
    with open(validation_instances_file_path, 'r') as validation_instances_file:
        instances_strings = validation_instances_file.readlines()
    validation_instances = []
    # Parse the strings in each line as a list
    for instance_str in instances_strings:
        validation_instances.append(ast.literal_eval(instance_str))
    return validation_instances


if __name__ == '__main__':
    __naive_bayes_parser('iris/iris.data', 0.8)  # this is commented because it's not necessary to run it again
    __naive_bayes_parser('covtype/covtype.data', 0.8) # this is commented because it's not necessary to run it again
