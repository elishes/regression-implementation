from sklearn.model_selection import train_test_split
import importlib
import my_model_tree
import numpy as np
import csv
import pandas as pd
import re


def main():
    
    #get input from user
    file_path_X = input('Full path of data file X: \n')
    file_path_y = input('Full path of data file y: \n')
    model_str = input("Insert full name of linear regression model you wish to run: \n")
    min_samples_split = int(input("Please choose minimum number of instances to split a node(integer): \n"))


    # Load data and model
    with open(file_path_X) as file_X:
        X = []
        
        for line in csv.reader(file_X):
            newline_X = []
            for element in get_line(line):
                try:
                    felement = float(element)
                    newline_X.append(felement)
                except:
                    newline_X.append(element) 
            X.append(newline_X)
           
    X = np.array(X, dtype="O")

    with open(file_path_y) as file_y:
        y = [float(el.strip()) for el in get_line(file_y.readlines()) if el]   
    y = np.array(y, dtype="O")


    
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    Modelattr = getattr(importlib.import_module("sklearn.linear_model"), model_str)
    model = Modelattr()
    
    
    # Build model tree
    model_tree = my_model_tree.MyModelTree(model, min_samples_split)
    
    # Train model tree and predict y_pred
    model_tree.fit(X_train, y_train)
    y_pred, error = model_tree.tree_predict(X_test, y_test)
    print(error) #prints MSE to terminal
    

    # Export y_pred as csv
    y_pred_2d = y_pred.reshape(-1, 1)
    y_pred_df = pd.DataFrame(y_pred_2d)
    
    y_pred_df.to_csv("model tree predictions")

def get_line(line):
    return [re.sub(r'[^\x00-\x7F]', "", el) for el in line]
                    
if __name__ == '__main__':
    main()

