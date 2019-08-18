# Laboratorio 3 del curso Aprendizaje Automático 2019.
Se implementan dos clasificadores:
- un clasificador basado en el algoritmo K-Nearest Neighbour.
- el clasificador Naive Bayes, con capacidad para manejar atributos numéricos.

## Dependencias
* python3 >= 3.5.2
* kdtree >= 0.16 : https://github.com/stefankoegl/kdtree

Para instalar la última versión de `kdtree` como dependencia de python3 ejecutar `pip3 install [--user] kdtree`.

También, se utilizan paquetes que por defecto vienen con Python: ast, json, math, statistics.

## Modos de invocación
Hay dos modos de uso: *NB* y *KNN*. El primero llama al clasificador *Naive Bayes* y el segundo al clasificador *K-Nearest Neighbour*. En ambos casos, se puede indicar que data set usar (iris o covtype). En el caso de *KNN*, se puede indicar el valor de *k* que se desea utilizar, que debe ser un natural mayor a 1.

### Evaluar un clasificador
#### KNN
Para Evaluar el algoritmo de *K-Nearest Neighbour*, invocar como:

python3 Main.py [kNN] [iris|covtype] [k]

#### NB
Para Evaluar el algoritmo de *Naive Bayes*, invocar como:

python3 Main.py [NB] [iris|covtype] [k]

## Archivos generados
Los resultados de los experimentos se guardan en dos directorios distintos según el clasificador utilizado. Los resultados de *KNN* se guardan en el directorio `knn_exp`, mientras que los de *NB* se guardan en `naive_bayes_exp`.
Luego de clasificar las instancias con *KNN*, el directorio `knn_exp` queda como sigue

```
+-- knn_exp
|   +-- covtype1.data
|   +-- covtype3.data
|   +-- ...
|   +-- iris1.data
|   +-- iris3.data
|   +-- ...
```
donde irisn.data/covtypen.data es el resultado de clasificar instancias del data set iris/covtype utilizando el clasificador *KNN* con *K=n*.
Luego de clasificar las instancias con *NB*, el directorio `naive_bayes_exp` queda como sigue

```
+-- naive_bayes_exp
|   +-- covtype.data
|   +-- iris.data
```
donde iris.data/covtype.data es el resultado de clasificar instancias del data set iris/covtype utilizando el clasificador *NB*.

En todos los archivos de resultados, la información que se guarda es, en primer lugar, la matriz de confusión; en segundo lugar, toda la información relacionada a las métricas del algoritmo, desagregada por clase:
- verdaderos positivos
- falsos positivos
- falsos negativos
- verdaderos negativos
- precisión
- recuperación
- fall-off
- F-Measure
Por último, se guarda también las medidas Macro y Micro de todas las métricas mencionadas antes.
