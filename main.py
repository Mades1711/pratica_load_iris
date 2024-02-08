import pandas as pd 
import numpy as np 
import pickle
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


#Extract data
def extract_data():
    data = load_iris()
    return data



#Preparing features
def preparing_features(data):
    x = pd.DataFrame(data.data, columns = data.feature_names)
    y = pd.DataFrame(data.target, columns=["target"])
    return x,y


# Train model
def train_model(x, y):
    model = DecisionTreeClassifier(max_depth=2) #quantidade de 'galhos' da arvore
    model.fit(x,y)
    return model


# Serialize object
def serialize_object(model):
    with open('trained_classifier.pkl','wb') as file:
        pickle.dump(model, file) #salvando o estado atual do modelo treinado


# Run training pipeline
def run():
    data = extract_data()
    x,y = preparing_features(data=data)
    model = train_model(x,y)
    serialize_object(model=model)

if __name__ == "__main__":
    run()