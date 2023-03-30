from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Cargamos el conjunto de datos IrisPlant
iris = load_iris()

# Definimos los datos y etiquetas de destino
X, Y = iris.data, iris.target

# Creamos las estructuras de datos de las métricas
exactitud1 = []
sensibilidad1 = []
precision1 = []
f1_scores1 = []

exactitud4 = []
sensibilidad4 = []
precision4 = []
f1_scores4 = []

exactitud7 = []
sensibilidad7 = []
precision7 = []
f1_scores7 = []

# Creamos una instancia de la clase StratifiedKFold, para realizar la validación cruzada
skf = StratifiedKFold(n_splits=10)

# Iteramos a través de cada división y ajustar los clasificadores KNN
for train_index, test_index in skf.split(X, Y):
    # Dividimos los datos en conjuntos de entrenamiento y prueba
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    # Creamos los clasificadores kNN con valores 1, 4 y 7
    knn1 = KNeighborsClassifier(n_neighbors=1)
    knn4 = KNeighborsClassifier(n_neighbors=4)
    knn7 = KNeighborsClassifier(n_neighbors=7)

    # Ajustamos los clasificadores KNN a los datos de entrenamiento
    knn1.fit(X_train, Y_train)
    knn4.fit(X_train, Y_train)
    knn7.fit(X_train, Y_train)

    # Realizamos predicciones en los datos de prueba
    Y_pred1 = knn1.predict(X_test)
    Y_pred4 = knn4.predict(X_test)
    Y_pred7 = knn7.predict(X_test)

    # Calculamos las métricas de evaluación y las agregamos a las listas de resultados
    exactitud1.append(accuracy_score(Y_test, Y_pred1))
    precision1.append(precision_score(Y_test, Y_pred1, average='weighted'))
    sensibilidad1.append(recall_score(Y_test, Y_pred1, average='weighted'))
    f1_scores1.append(f1_score(Y_test, Y_pred1, average='weighted'))

    exactitud4.append(accuracy_score(Y_test, Y_pred4))
    precision4.append(precision_score(Y_test, Y_pred4, average='weighted'))
    sensibilidad4.append(recall_score(Y_test, Y_pred4, average='weighted'))
    f1_scores4.append(f1_score(Y_test, Y_pred4, average='weighted'))

    exactitud7.append(accuracy_score(Y_test, Y_pred7))
    precision7.append(precision_score(Y_test, Y_pred7, average='weighted'))
    sensibilidad7.append(recall_score(Y_test, Y_pred7, average='weighted'))
    f1_scores7.append(f1_score(Y_test, Y_pred7, average='weighted'))

# Imprimimos las métricas de evaluación promedio para cada k
print("Metricas para k=1\n")
print("  Exactitud promedio:", sum(exactitud1) / 10)
print("  Sensibilidad promedio:", sum(sensibilidad1) / 10)
print("  Precisión promedio:", sum(precision1) / 10)
print("  F1-score promedio:", sum(f1_scores1) / 10)

print("\nMetricas para k=4\n")
print("  Exactitud promedio:", sum(exactitud4) / 10)
print("  Sensibilidad promedio:", sum(sensibilidad4) / 10)
print("  Precisión promedio:", sum(precision4) / 10)
print("  F1-score promedio:", sum(f1_scores4) / 10)

print("\nMetricas para k=7\n")
print("  Exactitud promedio:", sum(exactitud7) / 10)
print("  Sensibilidad promedio:", sum(sensibilidad7) / 10)
print("  Precisión promedio:", sum(precision7) / 10)
print("  F1-score promedio:", sum(f1_scores7) / 10)