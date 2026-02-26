import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np

## Leitura das Plantas
planta1 = pd.read_csv('plantsMachine.csv')
planta2 = pd.read_csv('plants2Machine.csv')

## CSV para um dataframe com a coluna "regada"
planta1['regada'] = 1
planta2['regada'] = 0
df = pd.concat([planta1, planta2], ignore_index=True)


#Preparacao Dados para Modelos
colunas = ["light", "temperature", "moisture", "conductivity"]
X = df[colunas]
y = df['regada']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=42)

def calculoClassifacao (luz,temperatura,humidade,fertilidade) :
    new_data = pd.DataFrame([[luz, temperatura, humidade, fertilidade]], columns=colunas)
    modeloDecisionTree = DecisionTreeClassifier()
    modeloDecisionTree.fit(X_train, y_train)
    return modeloDecisionTree.predict(new_data)


# Funcao que recebe modelo da Machine Learning e guarda o grafico e o csv
def aplicar_modelo(modelo, nomeModelo) :
    pred_all = modelo.predict(X)
    fig, axs = plt.subplots(len(colunas), 1, figsize=(8, 12))
    for i, coluna in enumerate(colunas):
        colors = np.where(pred_all == 0, 'red', 'blue')
        axs[i].bar(range(len(df)), df[coluna], color=colors)
        axs[i].set_ylabel(coluna)
        axs[i].set_xlabel('Data')
        axs[i].set_xticks(df.index)
        axs[i].set_xticklabels(df['registDay'], rotation=45, ha='right', fontsize = 8)
    fig.suptitle(nomeModelo, fontsize=16)
    plt.tight_layout()
    plt.savefig("graficos/" + nomeModelo + ".png")
    df_modelo = df.drop('regada', axis=1)
    df_modelo['Growth'] = pred_all
    df_modelo.to_csv('csvs/plants' + nomeModelo + ".csv", index=False)


if __name__ == "__main__":
    
    modeloDecisionTree = DecisionTreeClassifier()
    modeloDecisionTree.fit(X_train, y_train)
    aplicar_modelo(modeloDecisionTree, "DecisionTree")

    # Modelo RandomForest
    modeloRandom = RandomForestClassifier()
    modeloRandom.fit(X_train, y_train)
    aplicar_modelo(modeloRandom, "RandomForest")

    # Modelo SVM
    modeloSVM = SVC()
    modeloSVM.fit(X_train, y_train)
    aplicar_modelo(modeloSVM, "SupportVectorMachine")

    # Modelo Nearest Neighbors
    modeloNearest = KNeighborsClassifier()
    modeloNearest.fit(X_train, y_train)
    aplicar_modelo(modeloNearest, "NearestNeighbors")

    # Modelo Guassian
    modeloGuassian = GaussianNB()
    modeloGuassian.fit(X_train, y_train)
    aplicar_modelo(modeloGuassian, "GuassianNaiveBayes")

    # Modelo GradientBoosting
    modeloGradient = GradientBoostingClassifier()
    modeloGradient.fit(X_train, y_train)
    aplicar_modelo(modeloGradient, "GradientTreeBoosting")

    # Modelo MLP (Redes Neurais)
    modeloMLP = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=42)
    modeloMLP.fit(X_train, y_train)
    aplicar_modelo(modeloMLP, "Redes Neurais MLP")
