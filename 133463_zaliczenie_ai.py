from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Wczytanie danych
wine_data = load_wine()
X_wine = wine_data.data
y_wine = wine_data.target

# Podział danych na zbiór treningowy, walidacyjny i testowy
X_train_val, X_test, y_train_val, y_test = train_test_split(X_wine, y_wine, test_size=0.3)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.3)

# KNeighborsClassifier
knn_classifier = KNeighborsClassifier()
param_grid_knn = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
grid_search_knn = GridSearchCV(knn_classifier, param_grid_knn, scoring='accuracy')
grid_search_knn.fit(X_train, y_train)
y_pred_knn = grid_search_knn.predict(X_val)
accuracy_knn = accuracy_score(y_val, y_pred_knn)

print("Accuracy dla KNeighborsClassifier:", accuracy_knn)
print("Najlepsze parametry dla KNeighborsClassifier:", grid_search_knn.best_params_)

# SVC
svc_classifier = SVC()
param_grid_svc = {'C': [1, 10, 100], 'kernel': ['linear', 'rbf']}
grid_search_svc = GridSearchCV(svc_classifier, param_grid_svc, scoring='accuracy')
grid_search_svc.fit(X_train, y_train)
y_pred_svc = grid_search_svc.predict(X_val)
accuracy_svc = accuracy_score(y_val, y_pred_svc)

print("Accuracy dla SVC:", accuracy_svc)
print("Najlepsze parametry dla SVC:", grid_search_svc.best_params_)

# RandomForestClassifier
rf_classifier = RandomForestClassifier()
param_grid_rf = {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10]}
grid_search_rf = GridSearchCV(rf_classifier, param_grid_rf, scoring='accuracy')
grid_search_rf.fit(X_train, y_train)
y_pred_rf = grid_search_rf.predict(X_val)
accuracy_rf = accuracy_score(y_val, y_pred_rf)

print("Accuracy dla RandomForestClassifier:", accuracy_rf)
print("Najlepsze parametry dla RandomForestClassifier:", grid_search_rf.best_params_)

# Wybór najlepszego modelu na podstawie wyników walidacji krzyżowej
best_model = None
best_accuracy = 0.0

if accuracy_rf >= accuracy_knn and accuracy_rf >= accuracy_svc:
    best_model = grid_search_rf
    best_accuracy = accuracy_rf
elif accuracy_knn >= accuracy_rf and accuracy_knn >= accuracy_svc:
    best_model = grid_search_knn
    best_accuracy = accuracy_knn
else:
    best_model = grid_search_svc
    best_accuracy = accuracy_svc

print("Najlepszy model (z wyników walidacji krzyżowej):", best_model.estimator)
print("Najlepsza dokładność (z wyników walidacji krzyżowej):", best_accuracy)

# Testowanie najlepszego modelu na zbiorze testowym
y_pred_best = best_model.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)

print("Dokładność (Accuracy) dla najlepszego modelu na zbiorze testowym:", accuracy_best)
