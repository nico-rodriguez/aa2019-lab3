"""
Read the data sets Iris and Cover type and process the instances, so that it can be used by the algorithms.

For K-NN:
+ normalize the attributes (assuming normal distribution).
+ leave unchanged the binary attributes of Cover type (OneHot).
+ remove the class label of each instance. Manage a hash table that for each instance (without class label) returns
the associated class label.
+ return the k-d-tree which contains all the instances.
"""

import ast
import json
import kdtree
import random
import Utils


"""
Parsers for K-NN algorithm.
"""

knn_directory = 'knn/'
knn_iris_data_file_name = knn_directory + 'iris_data.json'
knn_iris_processed_data_file_name = knn_directory + 'iris_processed_data.data'
knn_iris_validation_file_name = knn_directory + 'iris_validation.data'
knn_covtype_data_file_name = knn_directory + 'covtype_data.json'
knn_covtype_processed_data_file_name = knn_directory + 'covtype_processed_data.data'
knn_covtype_validation_file_name = knn_directory + 'covtype_validation.data'


# Parse the instances
# Create a dictionary: 
# dict  -> attributes_count (contains the counts of attributes)
#       -> class_count (contains the amount of different classes)
#       -> data set (contains the tuples of the data set)
# Save the dictionary in knn_data (json) and return it.
def __knn_parse_instances(data_set_file_path):
    # Parse each instance line to a list
    with open(data_set_file_path, 'r') as data_set_file:
        instances_lines = data_set_file.readlines()
    # Get the amount of classes
    if 'iris' in data_set_file_path:
        classes_number = 3
    elif 'covtype' in data_set_file_path:
        classes_number = 7
    else:
        raise Exception('Parser.knn_parse_binary_attributes: "iris" or "covtype" not present in data set file' +
                        ' path')
    # Initialize empty dictionary
    data = {}
    dataset = []
    # Add each line to the data set
    for instance_line in instances_lines:
        instance_tokens = instance_line.split(',')
        class_label = Utils.num(instance_tokens[-1])
        # Adjust the class label range of cover type from [1,...,7] to [0,...,6]
        class_label = class_label if 'iris' in data_set_file_path else class_label - 1
        instance_tokens[-1] = class_label
        # Change the attributes from str to its current type
        for instance_index in range(len(instance_tokens)-1):
            instance_tokens[instance_index] = Utils.num(instance_tokens[instance_index])
        dataset.append(instance_tokens)
    data["dataset"] = dataset
    data["attributes_count"] = len(dataset[0])-1
    data["class_count"] = classes_number

    # Save the instances dictionary as a json file
    if 'iris' in data_set_file_path:
        instances_file_name = knn_iris_data_file_name
    else:
        instances_file_name = knn_covtype_data_file_name
    with open(instances_file_name, 'w') as knn_instances_file:
        json.dump(data, knn_instances_file)

    return data


# Loads the dictionary with the related data. Returns the dictionary.
def __knn_load_instances(instances_file_path):
    with open(instances_file_path, 'r') as instances_file:
        instances_per_class_string_keys = json.load(instances_file)
    return instances_per_class_string_keys


# Returns the minimum and maximum value of an attribute on the data set as a tuple (min, max)
def __knn_range_attribute_value(data, attribute):
    minimum_attribute_value = data["dataset"][0][attribute]
    maximum_attribute_value = minimum_attribute_value
    for entry in data["dataset"]:
        if entry[attribute] < minimum_attribute_value:
            minimum_attribute_value = entry[attribute]
        if entry[attribute] > maximum_attribute_value:
            maximum_attribute_value = entry[attribute]
    return minimum_attribute_value, maximum_attribute_value


# Receives parsed data.
# Normalizes the attributes (using re scale).
# Leaves unchanged the binary attributes of Cover type (OneHot).
# Removes the class label of each instance. Manage a hash table that for each instance (without class label) returns
# the associated class label.
# 'training_proportion' is the proportion of instances used for training. The remaining are used for validation.
# Saves the instances for training and validation normalized.
def __knn_transform_data_set(data, training_proportion):
    # Normalize the attributes
    for attribute in range(data["attributes_count"]):
        (minimum, maximum) = __knn_range_attribute_value(data, attribute)
        for entry in data["dataset"]:
            if maximum is not minimum:
                entry[attribute] = (entry[attribute] - minimum) / (maximum - minimum)

    # Save instances for validation
    if data['attributes_count'] == 4:
        validation_data_file_name = knn_iris_validation_file_name
    else:
        validation_data_file_name = knn_covtype_validation_file_name
    with open(validation_data_file_name, 'w') as validation_data_file:
        random.shuffle(data['dataset'])
        validation_instances_number = int(len(data['dataset']) - len(data['dataset']) * training_proportion)
        validation_instances = data['dataset'][0:validation_instances_number]
        data['dataset'] = data['dataset'][validation_instances_number:]
        for validation_instance in validation_instances:
            validation_data_file.write(str(validation_instance) + '\n')

    # Save the processed data and the dictionary to files
    if data['attributes_count'] == 4:
        processed_data_file_name = knn_iris_processed_data_file_name
    else:
        processed_data_file_name = knn_covtype_processed_data_file_name
    with open(processed_data_file_name, 'w') as processed_data_file:
        for instance in data['dataset']:
            processed_data_file.write(str(instance) + '\n')


# Removes the class label from the list of instances (last index) and returns a pair
# with the new data set and the hash table (dictionary)
def __knn_remove_class_label(data):
    dictionary = {}
    new_entries = []
    for entry in data:
        label = entry.pop(-1)
        dictionary[tuple(entry)] = label
        new_entries.append(tuple(entry))
    return new_entries, dictionary


def __knn_save_processed_data(dataset_file_path, training_proportion):
    data = __knn_load_instances(dataset_file_path)
    __knn_transform_data_set(data, training_proportion)


def knn_load_processed_data_and_dictionary(processed_data_file_path, validation_file_path):
    # Convert the lists in the processed_data_file_path to tuples
    dataset = []
    with open(processed_data_file_path, 'r') as processed_data_file:
        processed_data_lines = processed_data_file.readlines()
    for data_line in processed_data_lines:
        dataset.append(ast.literal_eval(data_line))
    classless_dataset, hash_ = __knn_remove_class_label(dataset)
    # Make the k-d-tree associated with the tuples on the data set
    tree = kdtree.create(classless_dataset)
    # Load the validation instances
    with open(validation_file_path, 'r') as validation_file:
        validation = []
        for validation_instance_line in validation_file.readlines():
            validation.append(ast.literal_eval(validation_instance_line))
    # Create the validation dictionary
    validation_dictionary = {}
    for validation_instance in validation:
        validation_dictionary[tuple(validation_instance[:-1])] = validation_instance[-1]
    return tree, hash_, validation_dictionary


if __name__ == '__main__':
    __knn_save_processed_data(knn_iris_data_file_name, 0.8)
    __knn_save_processed_data(knn_covtype_data_file_name, 0.8)
