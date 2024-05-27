from utils import db_connect
engine = db_connect()

# your code here
import os
import pandas as pd
import requests
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split

# 1. descargar data

url = "https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv"
respuesta = requests.get(url)
nombre_archivo = "diabetes.csv"
with open(nombre_archivo, 'wb') as archivo:
     archivo.write(respuesta.content)

# 2. convertir csv en dataframe

total_data = pd.read_csv("../data/raw/diabetes.csv")
# total_data.shape
# 768, 9

# borrar duplicados
total_data_sin = total_data.drop_duplicates()  
total_data_sin.shape
# 768, 9 = no hay duplicados

total_data.info()

# non null valores = 0
# todas las variables son numericas

ncols = total_data.columns.tolist()
print(ncols)

total_data_drop = total_data.drop(['SkinThickness'], axis=1)

# hay un error en los valores de "BloodPressure" = valores = 0 no son posibles

total_data_drop_fil = total_data_drop[(total_data_drop['BloodPressure'] != 0) ]
total_data_drop_fil.shape


fig, axis = plt.subplots(2, 4, figsize = (10, 7), gridspec_kw={'height_ratios': [1, 1]})

sns.histplot(ax = axis[0, 0], data = total_data_drop_fil, x = "Pregnancies").set(xlabel = "Pregnancies")
sns.histplot(ax = axis[0, 1], data = total_data_drop_fil, x = "Glucose").set(xlabel = "Glucose", ylabel = None)
sns.histplot(ax = axis[0, 2], data = total_data_drop_fil, x = "BloodPressure").set(xlabel = "BloodPressure", ylabel = None)
sns.histplot(ax = axis[0, 3], data = total_data_drop_fil, x = "Insulin").set(xlabel = "Insulin", ylabel = None)
sns.histplot(ax = axis[1, 0], data = total_data_drop_fil, x = "BMI").set(xlabel = "BMI", ylabel = None)
sns.histplot(ax = axis[1, 1], data = total_data_drop_fil, x = "DiabetesPedigreeFunction").set(xlabel = "DiabetesPedigreeFunction", ylabel = None)
sns.histplot(ax = axis[1, 2], data = total_data_drop_fil, x = "Age").set(xlabel = "Age", ylabel = None)
sns.histplot(ax = axis[1, 3], data = total_data_drop_fil, x = "Outcome").set(xlabel = "Outcome", ylabel = None)

plt.tight_layout()
plt.show()

# analisis de correlaciones

fig, axis = plt.subplots(figsize = (10, 6))

sns.heatmap(total_data[['Outcome', 'Pregnancies', 'Glucose', 'BloodPressure', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age' ]].corr(), annot = True, fmt = ".2f")
plt.tight_layout()
plt.show()

# bloodpresure tiene una corrralcion muy debil con el outcome asi que la podemos eliminar

total_data_eda = total_data_drop_fil.drop(['BloodPressure'], axis=1)
print(total_data_eda.head())

# corroboramos teoria vista en matriz de correlacion 

fig, axis = plt.subplots(figsize = (10, 5), ncols = 4)

sns.regplot(ax = axis[0], data = total_data_eda, x = "Age", y = "Outcome")
sns.regplot(ax = axis[1], data = total_data_eda, x = "BMI", y = "Outcome")
sns.regplot(ax = axis[2], data = total_data_eda, x = "Glucose", y = "Outcome")
sns.regplot(ax = axis[3], data = total_data_eda, x = "Pregnancies", y = "Outcome")

plt.tight_layout()
plt.show()

total_data_eda.describe()

# las avariables con mas outliers son DiabetesPedigreeFunction y Insulin, pero no son las mas importantes, asi que dejmaos los ouitliers

# split train y test

from sklearn.model_selection import train_test_split

# Dividimos el conjunto de datos en muestras de train y test
X = total_data.drop("Outcome", axis = 1)
y = total_data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

X_train.head()


# escalado de valores 

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train_norm = scaler.transform(X_train)
X_train_norm = pd.DataFrame(X_train_norm, index = X_train.index)

X_test_norm = scaler.transform(X_test)
X_test_norm = pd.DataFrame(X_test_norm, index = X_test.index)

X_train_norm["Outcome"] = list(y_train)
X_test_norm["Outcome"] = list(y_test)

X_train_norm.to_csv("../data/processed/clean_diabetes_train.csv", index=False)
X_test_norm.to_csv("../data/processed/clean_diabetes_test.csv", index=False)

# EDA COMPLETADO !!!!

train_df = pd.read_csv('../data/processed/clean_diabetes_train.csv')
print("Conjunto de datos de entrenamiento:")

test_df = pd.read_csv('../data/processed/clean_diabetes_test.csv')
print("\nConjunto de datos de prueba:")

X_train = train_df.drop(columns='Outcome')
y_train = train_df['Outcome']

X_test = test_df.drop(columns='Outcome')
y_test = test_df['Outcome']

# GRAFICO DE COORDENADAS PARALELAS
total_data = X
total_data["Name"] = y

pd.plotting.parallel_coordinates(total_data, "Name", color = ("#E58139", "#39E581", "#ffff00","#ff0000"))



# TEST ARBOL DE REGRESION
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(random_state = 42)
model.fit(X_train, y_train)

# GRAFICO DE ARBOL 
fig = plt.figure(figsize=(15,15))
tree.plot_tree(model, feature_names = list(X_train.columns), class_names = ["Pregnancies" , "Glucose", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"], filled = True)
plt.show()

# PREDICION 
y_pred = model.predict(X_test)
y_pred

# METRICA
from sklearn.metrics import mean_squared_error

print(f"Error cuadrático medio: {mean_squared_error(y_test, y_pred)}")

# grabar data
from pickle import dump

dump(model, open("decision_tree_regressor_default_42.sav", "wb"))


# TEST ARBOL DE CLASIFICACION

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state = 42)
model.fit(X_train, y_train)

from sklearn import tree

fig = plt.figure(figsize=(15,15))
tree.plot_tree(model, feature_names = list(X_train.columns), class_names = ["0", "1", "2"], filled = True)
plt.show()

y_pred = model.predict(X_test)
y_pred

from sklearn.metrics import accuracy_score, classification_report

print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

dump(model, open("decision_tree_classifier_default_42.sav", "wb"))

# OPTIMIZACION CLASSIFIER TREE

best_model = DecisionTreeClassifier(
    criterion='entropy', 
    max_depth= 5, 
    max_features= 4, 
    min_samples_leaf= 3, 
    min_samples_split=15,
    random_state=40
)

best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Precisión en el conjunto de prueba:", accuracy)
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

dump(model, open("decision_tree_classifier_tweaked_40.sav", "wb"))

# modelo mejorado en accuracy de 0.747 a 0.785