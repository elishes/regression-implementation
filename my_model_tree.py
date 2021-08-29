
"""
 my_model_tree.py
 
"""
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

class MyModelTree(object):

    def __init__(self, model, min_samples_split):
        self.model = model
        self.min_samples_split = min_samples_split
        self.tree = None

    def get_params(self):
        return {
            "model": self.model,
            "min_samples_split": self.min_samples_split
        }


    # ======================
    # Fit
    # ======================
    def fit(self, X, y):

        # Settings
        model = self.model
        min_samples_split = self.min_samples_split
        
        def _build_tree(X, y):

            global index_node_global
            # bigger thinking - the roll of the loss and model nodes
            def _create_node(X, y, container):
                data = (X, y)
                loss_node_model_node = fit_model(data, model)
                node = {"name": "node",
                        "index": container["index_node_global"],
                        "loss": loss_node_model_node[0],
                        "model": loss_node_model_node[1],
                        "data": data,
                        "n_samples": len(X),
                        "feature": None, 
                        "threshold": None, 
                        "is_num_split": None,
                        "children": {}
                        }
                container["index_node_global"] += 1
                return node

            # Split and traverse nodes until a terminal node is reached (recursive)
            def _split_traverse_node(node, container):
                if node["n_samples"] < min_samples_split:
                    return
                else:
                # Perform split and collect result
                    result = splitter(node, model,min_samples_split=min_samples_split)
                    if not result["did_split"]:
                        return
    
                    # Update node information based on splitting result
                    node["feature"] = result["feature"]
                    node["threshold"] = result["threshold"]
                    node["is_num_split"] = result["is_num_split"]
                    del node["data"]  # delete node stored data
    
                    # Extract splitting results
                    i = 0
                    if node["is_num_split"] == True: #for num split
                        for data_piece in result["data"]:
                            X, y = data_piece
                            node["children"][i] = _create_node(X, y, container)
                            node["children"][i]["model"] = result["models"][i]
                            i+=1
                    else: #for categorical split
                        for data_piece in result["data"]:
                            X, y = data_piece
                            node_key = node["threshold"][i]
                            node["children"][node_key] = _create_node(X, y, container)
                            node["children"][node_key]["model"] = result["models"][i]
                            i+=1
    
                    # Split nodes
                    list_of_children = node["children"].keys()
                    for child in list_of_children:
                        _split_traverse_node(node["children"][child], container)
    
            container = {"index_node_global": 0}  # mutable container
            root = _create_node(X, y, container)  # root node
            _split_traverse_node(root, container)  # split and traverse root node

            return root

        # Construct tree
        self.tree = _build_tree(X, y)

    # ======================
    # Predict
    # ======================
    def tree_predict(self, X, y):
        assert self.tree is not None       
        i = 0
        X_hot = encode_hotly(X) # encode cat variables to dummies
        y_pred_list = []
        for x in X:
            y_pred_list.append(_predict(self.tree, x, i, X_hot))  
            i+=1
        y_pred = np.array(y_pred_list)
        y_pred_error = loss(y, y_pred)
        
        return y_pred, y_pred_error

        

# ***********************************
#
# internal functions
#
# ***********************************

    
def splitter(node, model,
              min_samples_split):
    
    # Extract data
    X, y = node["data"]
    N, d = X.shape

    # Find feature splits that might improve loss
    temp_result = {"did_split": False,
               "loss_best": node["loss"],
                "data_best": None,
                "models_best": None,
                "feature_best": None,
                "threshold_best": None,
                "is_num_split": None}
    
    result = {"did_split": False,
                  "loss": node["loss"],
                  "models": node["model"],
                  "data": node["data"],
                  "feature": node["feature"],
                  "threshold": node["threshold"],
                  "is_num_split": node["is_num_split"],
                  "n_samples": N}
     
    for feature in range(d):
     # generate threshold search list (feature)
        threshold_search = X[:, feature].tolist()
        threshold_search_better = []
        for value in threshold_search:
            try:
                value = float(value)
            except:
                pass
            threshold_search_better.append(value)
        threshold_search = threshold_search_better            
        
        if len(set(threshold_search)) == 1:
            continue        
        else:
            is_num_split = True
            
            for threshold in threshold_search:
                if type(threshold) == float:     # find split for num variable
                    data_chunks, z_weights = split_data_binary(node, feature,
                                                               threshold, X, y)
                    temp_result = find_best_split(temp_result, node,
                                                  data_chunks, feature, z_weights,
                                                  model, is_num_split)                
                else: #find aplit for cat variable
                    data_chunks, z_weights = split_data_cat(node, feature, X, y)
                    is_num_split = False
                    temp_result = find_best_split(temp_result, node, data_chunks,
                                                  feature, z_weights, model,
                                                  is_num_split)
            
                    
        # Return the best result
        result = {"did_split": temp_result["did_split"],
                  "loss": temp_result["loss_best"],
                  "models": temp_result["models_best"],
                  "data": temp_result["data_best"],
                  "feature": temp_result["feature_best"],
                  "threshold": temp_result["threshold_best"],
                  "is_num_split": temp_result["is_num_split"],
                  "n_samples": N}
    
    return result

def split_data_binary(node, feature, threshold, X, y): # binary numeric split
    idx_left = np.where(X[:, feature] <= threshold)[0]
    idx_right = np.delete(np.arange(0, len(X)), idx_left)
    assert len(idx_left) + len(idx_right) == len(X)
    return [(X[idx_left], y[idx_left]), (X[idx_right], y[idx_right])],[(threshold, len(idx_left)), (threshold, len(idx_right))]

def split_data_cat(node, feature, X, y):# categorical multisplit
    features_unique, weights = np.unique(X[:,feature], return_counts = True)
    indices = []
    data_chunks = []
    
    for i in features_unique:
        indices.append(np.where(X[:, feature] == i)) 
    
    for j in indices:
        data_chunks.append((X[j], y[j]))
    
    zipped_weights = zip(features_unique, weights)
    return data_chunks, zipped_weights

def loss(y, y_pred): #calculate MSE
    
    if y_pred.ndim > 2:
        y_pred = y_pred[:,:,0]
    
    return mean_squared_error(y, y_pred)

def fit_model(data, model):
    X = data[0]
    y = data[1]
    if X.ndim == 1:
        X = X.reshape(1, -1)
    N_reg, d_reg = X.shape
    
    if N_reg == 0:
        return 0    
    else:
        X_hot = encode_hotly(X)
        X_hot = X_hot.astype(np.float64)
        N_hot, d_hot = X_hot.shape
        if N_hot == 0:
            return 0        
        else:
            model.fit(X_hot, y)
            y_pred = model.predict(X_hot)
            model_loss = loss(y, y_pred)
            assert model_loss >= 0.0
            return [model_loss, model]
    
def find_best_split(temp_result, node, data_chunks, feature, zipped_weights,
                    model, is_num_split):
    MSE_model_list = []
    data_chunks_new = data_chunks
    
    for data in data_chunks:
        new_mnm = fit_model(data, model)        
        if new_mnm == 0:
            data_chunks_new.remove(data)
            continue        
        else:
            MSE_model_list.append(fit_model(data, model))
    
    MSE_model = np.array(MSE_model_list, dtype="O")
    MSEs = MSE_model[:, 0]
    models = MSE_model[:,1]
    features_unique, weights = zip(*zipped_weights)
    loss_split = sum(weights*MSEs/node["n_samples"])    

    # Update best parameters if loss is lower
    if loss_split < temp_result["loss_best"]:
        temp_result["did_split"] = True
        temp_result["loss_best"] = loss_split
        temp_result["models_best"] = models
        temp_result["data_best"] = data_chunks_new
        temp_result["feature_best"] = feature
        temp_result["is_num_split"] = is_num_split
        temp_result["threshold_best"] = features_unique
    
    return temp_result

def encode_hotly(X): #encodes cat variables into dummy variables
    X_cat_list = []
    X_num_list = []
    enc = OneHotEncoder(handle_unknown='ignore')
    
    N, d = X.shape
    
    if N == 0:
        return 0    
    for i in range(d):
        try:
            float(X[0,i])
            X_num_list.append(X[:,i])
        except:
            X_cat_list.append(X[:,i])            
            
    X_cat = np.array(X_cat_list, dtype = "O")
    X_num = np.array(X_num_list, dtype = "O")
    
    #if X_ndim != 1:
    X_cat = X_cat.transpose()
    X_num = X_num.transpose()
    
    if len(X_cat) == 0:
        return X_num        

    X_cat_hotly = enc.fit_transform(X_cat).toarray()

    if len(X_num) == 0:
        return X_cat_hotly        
    elif X_num.ndim == 1:
        X_num = X_num.reshape(1,-1)

    X_all_hotly = np.concatenate((X_cat_hotly, X_num), axis=1)
    
    return X_all_hotly

def _predict(node, x, row, X_hot): 
#recursively runs each row through nodes until termina, then runs model
    x_hot = X_hot[row,:]
    x = x.reshape(1,-1)
    x_hot = x_hot.reshape(1,-1)
    if node["children"] == {}:            
        y_pred_x = node["model"].predict(x_hot)
        return y_pred_x
    else:
        if node["is_num_split"] == True:
            if x[0, node["feature"]] <= node["threshold"][0]:
                return _predict(node["children"][0], x, row, X_hot)
            else:
                return _predict(node["children"][1], x, row, X_hot)
        else:
            if x[0, node["feature"]] in node["threshold"]:
                key=x[0, node["feature"]]
                return _predict(node["children"][key], x, row, X_hot)
            else:
                del x[0, node["feature"]]
                x_hot2 = encode_hotly(x)
                y_pred_x = node["model"].predict(x_hot2)
                return y_pred_x