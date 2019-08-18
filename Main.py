'''
Main module of the project.
Assumes, data has been already preprocessed.
TODO: data shouldnt be preprocessed if using [trianing] param. should it be removed?
'''

import sys
import KNN
import NaiveBayes
import KNNParser
import NBParser
import Evaluator

uso_general = """
Invocar como:

python3 Main.py [kNN|NB] [iris|covtype] [k]

donde:

- kNN o NB indica el algoritmo a utilizar.
- iris o covtype indica el nombre del dataset que se utilizará.
- k, indica el valor de k a utilizar en el caso de kNN. Si se usa NB, 
se puede omitir. Debe ser un entero positivo. En caso de introducir
un real, se trunca y se toma su parte entera.
"""

if __name__ == "__main__":
    # checking arguments
    if (len(sys.argv) < 2):
        print('#########################')        
        print('Error. Cantidad de parametros invalidos')
        print('-------------------------')
        print(uso_general)
        exit()
    mode = sys.argv[1]
    dataset = sys.argv[2]
    if dataset != "iris" and dataset != "covtype":
        print('#########################')        
        print('Error. Dataset inválido. Debe ser iris o covtype')
        print('-------------------------')
        exit()
    if mode != "kNN" and mode != "NB":
        print('#########################')
        print('Error. Modo de uso inválido.')
        print('-------------------------')
        print(uso_general)
        exit()
    if mode == "kNN":
        k = int(sys.argv[3])
        if k < 0:
            print('#########################')
            print('Error. Valor de k negativo, o cero. Debe ser positivo')
            print('-------------------------')
            exit()
    print("Entrenando el dataset " + dataset + " con una proporción de entrenamiento del 0.8 con el algoritmo " + mode)
    if mode == "kNN":
        print("Tomando un valor de k de: " + str(k))
        if dataset == "iris":
            processed_data_file_name = KNNParser.knn_iris_processed_data_file_name
            validation_file_name = KNNParser.knn_iris_validation_file_name
            outputfile = 'knn_exp/iris{k}.data'.format(k=k)
        else: 
            processed_data_file_name = KNNParser.knn_covtype_processed_data_file_name
            validation_file_name = KNNParser.knn_covtype_validation_file_name
            outputfile = 'knn_exp/covtype{k}.data'.format(k=k)
        print('Los resultados estaran en ' + outputfile)
        k_d_tree, hash_, validation_dictionary = KNNParser.knn_load_processed_data_and_dictionary(
        processed_data_file_name, validation_file_name)
        classification = KNN.knn_classify_instance_set(validation_dictionary, k, k_d_tree, hash_, [0, 1, 2, 3, 4, 5, 6])
        print("Evaluando el clasificador sobre el conjunto de validación")
        Evaluator.evaluate_classifier(classification, [0, 1, 2, 3, 4, 5, 6], outputfile)
    else:
        if dataset == "iris":
            distributions_dictionary = NBParser.naive_bayes_load_distributions(NBParser.naive_bayes_iris_distributions_file_name)
            validation_set = NBParser.naive_bayes_load_validation_instances(NBParser.naive_bayes_iris_validation_instances_file_name)
            classes_labels = ["Iris Setosa", "Iris Versicolour", "Iris Virginica"]
            classes_distributions = {"Iris Setosa" : 1/3, "Iris Versicolour" : 1/3, "Iris Virginica" : 1/3}
            outputfile = 'naive_bayes_exp/iris.data'
        else:
            distributions_dictionary = NBParser.naive_bayes_load_distributions(NBParser.naive_bayes_covtype_distributions_file_name)
            validation_set = NBParser.naive_bayes_load_validation_instances(NBParser.naive_bayes_covtype_validation_instances_file_name)
            classes_labels = ["Spruce/Fir", "Lodgepole Pine", "Ponderosa Pine", "Cottonwood/Willow", "Aspen", "Douglas-fir", "Krummholz"]
            classes_distributions = {"Spruce/Fir" : 211840/581012, "Lodgepole Pine" : 283301/581012,
                                     "Ponderosa Pine" : 35754/581012, "Cottonwood/Willow" : 2747/581012,
                                    "Aspen" : 9493/581012, "Douglas-fir" : 17367/581012, "Krummholz" : 20510/581012}
            outputfile = 'naive_bayes_exp/covtype.data'            
        print('Los resultados estaran en ' + outputfile)
        classified_data = NaiveBayes.naive_bayes_classify_dataset(classes_distributions, distributions_dictionary, classes_labels, validation_set)
        print("Evaluando el clasificador sobre el conjunto de validación")
        Evaluator.evaluate_classifier(classified_data, classes_labels, outputfile)

        