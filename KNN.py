"""
This module implements the K-Nearest Neighbour algorithm. It assumes that instances don't have a class label at the
end. Instead, it receives a hash table such that for each instance, returns the associated class label. It also assumes
that the instance's attributes are normalized. It receives a k-d-tree for efficiently
calculating the k nearest neighbours and their distances.
The classification function of this module returns a class label.
"""

import Evaluator
import KNNParser


# Implement the K-Nearest Neighbour algorithm.
# - 'x' is the instance to classify.
# - 'k' is the number of neighbours considered. It may be 1, 3 or 7.
# - 'k_d_tree' is a k-d-tree that contains all instances for efficient neighbour searching.
# - 'classes' is a list with all class labels.
# - 'class_label' is a dictionary: instance -> class label.
# Return the class number that classifies x with the K-NN algorithm.
def __knn_classify_instance(x, true_class, k, k_d_tree, class_label, classes):
    # Check if k is 1, 3 or 7
    # if k != 1 and k != 3 and k != 7:
        # raise Exception('K-NN.knn: expected a k value of 1, 3 or 7')

    # search_knn receives k+1 because it includes x in the result list of neighbours.
    # neighbours_pairs_list is a list of paris (KDNode, distance).
    neighbours_pairs_list = k_d_tree.search_knn(x, k)

    # Dictionary: class number -> cumulative distance of nodes of that class to x
    cumulative_distance = {}
    for c in classes:
        cumulative_distance[c] = 0.0

    # Compute the cumulative distances of neighbours of each class to x.
    for pair in neighbours_pairs_list:
        # The actual point that is a neighbour of x.
        neighbour_point = pair[0].data
        neighbour_distance = pair[1]
        neighbour_class = class_label[neighbour_point]
        cumulative_distance[neighbour_class] += 1 / neighbour_distance

    # Find the class of neighbours with highest cumulative distance to x and classify x with that class number.
    # In case of tie (very unlikely), the algorithm prefers to take the first class number considered.
    classified_class = None  # The class label for classifying the instance x.
    max_cum_distance = None  # The highest cumulative distance computed.
    for c in cumulative_distance:
        if max_cum_distance is None or max_cum_distance < cumulative_distance[c]:
            classified_class = c
            max_cum_distance = cumulative_distance[c]

    # Classify x as being of the class with the highest cumulative distance.
    return true_class, classified_class


# Apply the K-Nearest Neighbour algorithm to each instance in 'instances'
# - 'instances' is a list of instances to classify.
# - 'k' is the number of neighbours considered. It may be 1, 3 or 7.
# - 'k_d_tree' is a k-d-tree that contains all instances for efficient neighbour searching.
# - 'classes' is a list with all class labels.
# - 'class_label' is a dictionary: instance -> class label.
# Return the input parameters of evaluate_classifier:
#       + a list of tuples (true class, classified class).
#       + class labels.
def knn_classify_instance_set(instances, k, k_d_tree, class_label, classes):
    # Classify each instance in 'instances' in parallel
    # return Parallel(n_jobs=1,
    #                 verbose=100)(delayed(__knn_classify_instance)(instance, instances[instance], k, k_d_tree,
    #                                                              class_label, classes) for instance in instances)
    k_d_tree.rebalance()
    classification = []
    for instance in instances:
        classification.append(__knn_classify_instance(instance, instances[instance], k, k_d_tree, class_label, classes))
    return classification


if __name__ == '__main__':
    """
    k_d_tree, hash_, validation_dictionary = KNNParser.knn_load_processed_data_and_dictionary(
        KNNParser.knn_iris_processed_data_file_name, KNNParser.knn_iris_validation_file_name)
    for k in [1, 3, 7]:
        classification = knn_classify_instance_set(validation_dictionary, k, k_d_tree, hash_, [0, 1, 2])
        Evaluator.evaluate_classifier(classification, [0, 1, 2], 'knn_exp/iris{k}.data'.format(k=k))
    """
    k_d_tree, hash_, validation_dictionary = KNNParser.knn_load_processed_data_and_dictionary(
        KNNParser.knn_covtype_processed_data_file_name, KNNParser.knn_covtype_validation_file_name)
    for k in [1, 3, 7]:
        classification = knn_classify_instance_set(validation_dictionary, k, k_d_tree, hash_, [0, 1, 2, 3, 4, 5, 6])
        Evaluator.evaluate_classifier(classification, [0, 1, 2, 3, 4, 5, 6], 'knn_exp/covtype{k}.data'.format(k=k))
