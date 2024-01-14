from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import  Dense, Dropout
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate


def representation_of_data():
    data = pd.read_csv(
    "data.csv", 
    header=0)

    # Usunięcie ostatniej kolumny (Unnamed: 32)
    data = data.iloc[:, :-1]

    # Usunięcie wszystkich wartości NaN
    data = data.dropna()

    # Usunięcie kolumny "diagnosis", zamiast stworzenie nowej -"cancer" z wartościami 0,1
    data['cancer'] = data['diagnosis'].apply(lambda x: '1' if x == 'M' else '0')
    data = data.drop(data.columns[1], axis=1)

    return data

def models(X_train,Y_train):
    # Regresja logistyczna
    from sklearn.linear_model import LogisticRegression
    log=LogisticRegression(random_state=0)
    log.fit(X_train,Y_train)
        
    # Drzewa decyzyjne 
    from sklearn.tree import DecisionTreeClassifier
    tree=DecisionTreeClassifier(random_state=0,criterion="entropy")
    tree.fit(X_train,Y_train)

    # Sieci neuronowe    
    neural = Sequential()
    neural.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))
    neural.add(Dropout(0.2))  
    neural.add(Dense(units=64, activation='relu'))
    neural.add(Dropout(0.5))
    neural.add(Dense(units=64, activation='relu'))
    neural.add(Dropout(0.5))
    neural.add(Dense(units=1, activation='sigmoid'))

    neural.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    neural.fit(X_train, Y_train, batch_size=128, epochs=15, validation_split=0.1, verbose=0)

    return log, tree, neural

# Generowania raportów modeli
def make_report(predictions, Y_test):
    # Regresja logistyczna
    print("\nLogistic Regression:")
    print(classification_report(Y_test,predictions[0]))

    # Drzewa decyzyjne
    print("\nDecision Tree:")
    print(classification_report(Y_test,predictions[1]))

    # Sieci neuronowe
    print("\nNeural Networks:")
    print(classification_report(Y_test, predictions[2]))

# Ewaluacja modeli
def evaluate_models(predictions, Y_test):
    results = []
    model_name = ['Logistic Regression', 'Decision Tree', 'Neural Networks']
    for i in range (0,3):
        test_results = [
            model_name[i] + " (Test)",
            round(accuracy_score(Y_test, predictions[i]), 2),
            round(precision_score(Y_test, predictions[i]), 2),
            round(recall_score(Y_test, predictions[i]), 2),
            round(f1_score(Y_test, predictions[i]), 2)
        ]
        results.extend([test_results, []])

    headers = ["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
    print(tabulate(results, headers=headers, tablefmt="pretty", colalign=("left", "left", "left", "left", "left")))

# Ewaluacja dokładności i predykcji binarnych
def evaluate_binary_predictions_and_accuracies(log, tree, model, X_test, Y_test):
    # Predykcje dla każdego z modeli
    log_predict = log.predict(X_test)
    tree_predict = tree.predict(X_test)
    neural_predict = model.predict(X_test, verbose=0)
    neural_predict = np.array(neural_predict).flatten()

    # Konwersja predykcji na binarne na podstawie progu
    threshold = 0.5
    log_binary = (log_predict > threshold).astype(int)
    tree_binary = (tree_predict > threshold).astype(int)
    neural_binary = (neural_predict > threshold).astype(int)
    neural_binary = np.array(neural_binary).flatten()

    predictions = [log_predict, tree_predict, neural_binary]

    accuracies = [accuracy_score(Y_test, log_predict),
                  accuracy_score(Y_test, tree_predict),
                  accuracy_score(Y_test, neural_binary)]
    
    binary_predictions = [log_binary, tree_binary, neural_binary]

    return predictions, accuracies, binary_predictions

# Wykresy z macierzami pomyłek
def plot_confusion_matrix(Y_test, binary_predictions):
    classifiers = ['Logistic Regression', 'Decision Tree', 'Neural Networks']
    plt.figure(figsize=(9, 3))
    for i, classifier in enumerate(classifiers):
        plt.subplot(1, 3, i + 1)

        cm = confusion_matrix(Y_test, binary_predictions[i])

        sns.heatmap(cm, annot=True, fmt='d', cmap='BuPu', xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix - {classifier}')

    plt.tight_layout()
    plt.show()

# Wykres porównania dokładności
def plot_accuracy_comparison(accuracies):
    methods = ['Logistic Regression', 'Decision Tree', 'Neural Networks']
    accuracies_percentage = [accuracy * 100 for accuracy in accuracies]
    plt.bar(methods, accuracies_percentage, color=['crimson', 'orchid', 'lightpink'])
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Comparison')
    plt.ylim([0, 100])
    plt.show()

def run():
    # Wczytanie i przetworzenie danych
    data = representation_of_data()
    print(data.shape[0])

    # Podział danych na X i Y
    X = data.iloc[:, :-1].values  
    Y = data.iloc[:, -1].values 

    # Kodowanie etykiet do postaci binarnej
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y)

    # Podział na zestaw testowy i treningowy
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)
    num_examples_train = X_train.shape[0]
    num_examples_test = X_test.shape[0]

    print(f"Liczba przykładów w zbiorze treningowym: {num_examples_train}")
    print(f"Liczba przykładów w zbiorze testowym: {num_examples_test}")

    # Skalowanie
    X_train=StandardScaler().fit_transform(X_train)
    X_test=StandardScaler().fit_transform(X_test)

    # Trenowanie modeli
    log, tree, model = models(X_train, Y_train)

    # Ewaluacja dokładności i predykcji binarnych dla każdego z modeli
    predictions, accuracies, binary_predictions = evaluate_binary_predictions_and_accuracies(log, tree, model, X_test, Y_test)

    # Ewaluacja modeli
    make_report(predictions, Y_test)
    evaluate_models(predictions, Y_test)

    # Wygenerowanie wykresów z macierzami pomyłek dla każdego z modeli
    plot_confusion_matrix(Y_test, binary_predictions)

    # Wygenerowanie wykresu porównania dokładności między modelami
    plot_accuracy_comparison(accuracies)

    

if __name__ == '__main__':
    run()