"""
Read the data sets Iris and Cover type and process the instances, so that it can be used by the algorithms.

For both Naive Bayes and K-NN:
+ convert Iris class from str to int. The mapping is Iris-setosa <-> 0, Iris-versicolor <-> 1 and Iris-virginica <-> 2.
"""

"""
Parser for both algorithms. It is run only once.
"""


# Parse the iris data set and change the class label from string to integer.
# It saves the modified data set in a file with the same name
def __parse_iris_class_label(data_set_file_path):
    with open(data_set_file_path, 'r') as data_set_file:
        # List with instances as strings
        lines = data_set_file.readlines()
        # List of parsed lines
        new_lines = []
        for line in lines:
            # Split attributes and class label (last value)
            line_tokens = line.split(',')
            class_label = line_tokens[-1]
            if class_label == 'Iris-setosa\n':
                line_tokens[-1] = '0\n'
            elif class_label == 'Iris-versicolor\n':
                line_tokens[-1] = '1\n'
            elif class_label == 'Iris-virginica\n':
                line_tokens[-1] = '2\n'
            else:
                raise Exception('KNNParser.py: error reading class label.')
            new_lines.append(','.join(line_tokens))

    with open(data_set_file_path, 'w') as data_set_file:
        # Write the numbers (attributes and class) in the file with the same format
        data_set_file.writelines(new_lines)
