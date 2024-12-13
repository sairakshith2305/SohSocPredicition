# -*- coding: utf-8 -*-
"""
Created on Mon May 23 19:18:26 2022

@author: home
"""

from flask import Flask,render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from datetime import datetime
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')
    
@app.route('/models')
def models():
    return render_template('main.html')
    
@app.route('/graph')
def graph():
    return render_template('graph.html')
    
@app.route('/aboutUs')
def about():
    return render_template('about.html')
    
    
@app.route('/report')
def report():
    return render_template('report.html')
      
    
@app.route('/graphforSOH')
def graphforSOH():
    return render_template('sohgraph.html')

    
@app.route('/graphforSOC')
def graphforSOC():
    return render_template('socgraph.html')


@app.route('/svrsoh')
def svrsoh(): 
    #SVR MODEL-SOH
    df = pd.read_csv("soh_1.csv")    
    #df.describe()
    X_data=df.drop(['soh'],axis=1)
    y_data=df['soh']
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X_data, y_data, test_size=0.2,random_state=1)
    x_test_orig=X_test1
  
    x_test_orig=x_test_orig.values.tolist()
    X_train1=X_train1.drop(['TimeStamp'],axis=1)
    X_test1=X_test1.drop(['TimeStamp'],axis=1)
   
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    X_train1 = sc_x.fit_transform(X_train1)
    X_test1 = sc_x.transform(X_test1)
    y_train1 = sc_y.fit_transform(np.array(y_train1).reshape(-1,1))
    y_test1 = sc_y.transform(np.array(y_test1).reshape(-1,1))
    regressor = SVR(kernel = 'rbf')
    regressor.fit(X_train1,y_train1)
    
    
    
    y_prediction1 =  regressor.predict(X_test1)
    
    
    y_prediction2 = sc_y.inverse_transform(np.array(y_prediction1).reshape(-1,1))
    actual_soh = sc_y.inverse_transform(np.array(y_test1).reshape(-1,1))
    
    sortto=x_test_orig

    for i in range(len(y_prediction2)):
        sortto[i].append(actual_soh[i][0])
        sortto[i].append(y_prediction2[i])
    
    #for i in range(len(sortto)):
     #       sortto[i][0]=datetime.strftime(datetime.strptime(sortto[i][0],'%Y-%m-%d %H:%M:%S.%f'),'%Y-%m-%d %H:%M:%S.%f')
            
    from operator import itemgetter
    outputlist=sorted(sortto,key=itemgetter(0))
    status=[]
    for i in range(len(outputlist)):
        if(outputlist[i][9][0]>0.75):
            status.append("Battery in good condition")
        else:
            status.append("battery needs to be replaced")
    print("status array : ",status)
            
    #return render_template('home.html',soh=actual_soh,data=x_test_orig,pred=y_prediction2,length=len(y_prediction2))
    return render_template('svrsoh.html',batterystatus=status,sortto=outputlist,length=len(sortto))
@app.route('/svrsoc')
def svrsoc(): 
    #SVR MODEL-SOC
    df = pd.read_csv("soc.csv")
    
    X_data=df.drop(['Relative State of Charge'],axis=1)
    y_data=df['Relative State of Charge']
        
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X_data, y_data, test_size=0.2,random_state = 1)
    X_testwithTimeStamp1=X_test1
    x_test_origSOC1=X_testwithTimeStamp1.values.tolist()
    X_train1=X_train1.drop(['TimeStamp'],axis=1)
    X_test1=X_test1.drop(['TimeStamp'],axis=1)
    
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    X_train1 = sc_x.fit_transform(X_train1)
    X_test1 = sc_x.transform(X_test1)
    y_train1 = sc_y.fit_transform(np.array(y_train1).reshape(-1,1))
    y_test1 = sc_y.transform(np.array(y_test1).reshape(-1,1))
    
    
    regressor = SVR(kernel = 'rbf')
    regressor.fit(X_train1,y_train1)
    y_prediction1 =  regressor.predict(X_test1)
    
    
    actual_soc = sc_y.inverse_transform(np.array(y_test1).reshape(-1,1))
    
    print(y_test1)



    y_prediction1 = sc_y.inverse_transform(np.array(y_prediction1).reshape(-1,1))
    print("ypred:",y_prediction1)
    sortto=x_test_origSOC1
    print(len(y_prediction1))
    print(len(x_test_origSOC1[0]))
    for i in range(len(y_prediction1)):
        sortto[i].append(actual_soc[i][0])
        sortto[i].append(y_prediction1[i])
    print(sortto)
    
    #for i in range(len(sortto)):
     #       sortto[i][0]=datetime.strftime(datetime.strptime(sortto[i][0],'%Y-%m-%d %H:%M:%S.%f'),'%Y-%m-%d %H:%M:%S.%f')
            
    from operator import itemgetter
    outputlist=sorted(sortto,key=itemgetter(0))
    print("out:::",outputlist)
    status=[]
    for i in range(len(outputlist)):
        if(outputlist[i][8]>25):
            status.append("Battery in good condition")
        else:
            status.append("battery needs to be charged")
    print("status array : ",status)
            
    return render_template('svrsoc.html',batterystatus=status,sortto=outputlist,length=len(sortto))
    
    #return render_template('home.html',soh=y_test1,data=x_test_origSOC1,pred=y_prediction1,length=len(y_prediction1))

#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

@app.route('/mlstmsoh')
def mlstmsoh():
    import pickle
    with open('mlstmsoh.txt','rb') as f:
        inv_y=pickle.load(f)
   
    with open('mlstmsohpred.txt','rb') as f:
        inv_yhat=pickle.load(f)
  
      
    with open('mlstmsohx.txt','rb') as f:
        x_data_orig=pickle.load(f)
    status=[] 
    for i in range(len(inv_yhat)):
        if(inv_yhat[i]>0.74):
            status.append("Battery in good condition")
        else:
            status.append("battery needs to be replaced")
    print("status array : ",status)
    return render_template('mlstmsoh.html',batterystatus=status,soh=inv_y,data=x_data_orig,pred=inv_yhat,length=len(inv_yhat))


@app.route('/mlstmsoc')
def mlstmsoc():
    import pickle
    with open('mlstmsoc.txt','rb') as f:
        inv_y=pickle.load(f)
   
    with open('mlstmsocpred.txt','rb') as f:
        inv_yhat=pickle.load(f)
  
      
    with open('mlstmsocx.txt','rb') as f:
        x_data_orig=pickle.load(f)
    
    status=[]
    for i in range(len(inv_yhat)):
        if(inv_yhat[i]>2.5):
            status.append("Battery in good condition")
        else:
            status.append("battery needs to be charged")
    print("status array : ",status)
    return render_template('mlstmsoc.html',batterystatus=status,soh=inv_y,data=x_data_orig,pred=inv_yhat,length=len(inv_yhat))





@app.route('/nnsoc')
def nnsoc():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import pyswarms as ps
    import random
    from pandas import DataFrame
    from pandas import concat
    from sklearn.preprocessing import MinMaxScaler
    import pickle
    from scipy.optimize import differential_evolution
    
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
        
    from pandas import read_csv
    dataset = read_csv('soc1.csv', header=0, index_col=0)
    values = dataset.values
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler_x = MinMaxScaler(feature_range=(0, 1))

    y = values[:, -1]
    x = values[:,:-1]
    df = read_csv('soc1.csv')
    X_data=df.drop(['Relative State of Charge'],axis=1)
    x_data_orig=X_data[1202:]
    x_data_orig=x_data_orig.values.tolist()
    
    x = scaler_x.fit_transform(x)
    y = scaler.fit_transform(np.array(y).reshape(-1,1))
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, shuffle = False)
    y_test = scaler.inverse_transform(y_test)
    
    with open('nnsoc.txt','rb') as f:
        y_pred=pickle.load(f)
    print("y_pred nnsoc",type(x_data_orig))
    print("y_pred nnsoc",x_data_orig)
    status=[]
    for i in range(len(y_pred)):
        if(y_pred[i]>25):
            status.append("Battery in good condition")
        else:
            status.append("battery needs to be charged")
    print("status array : ",status)
            
    return render_template('nnsoc.html',batterystatus=status,soh=y_test,data=x_data_orig,pred=y_pred,length=len(y_pred))

 
  

@app.route('/nnsoh')
def nnsoh():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import pyswarms as ps
    import random
    from pandas import DataFrame
    from pandas import concat
    from sklearn.preprocessing import MinMaxScaler
    
    from scipy.optimize import differential_evolution
    
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    
    Rand = 42
    random.seed(Rand)
    np.random.seed(Rand)
    
    from pandas import read_csv
   
    dataset = read_csv('soh_1.csv', header=0, index_col=0)
    values = dataset.values
    # ensure all data is float
    values = values.astype('float32')
    df = read_csv('soh_1.csv', header=0)
    X_data=df.drop(['soh'],axis=1)
    x_data_orig=X_data[100:]
    x_data_orig=X_data.values.tolist()
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    
    y = values[:, -1]
  
    x = values[:,:-1]

    x = scaler_x.fit_transform(x)
    
    y = scaler.fit_transform(np.array(y).reshape(-1,1))
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, shuffle = False)
    def cal(n_inputs, n_hidden, n_classes):
        i_weights = n_inputs*n_hidden
        i_bias = n_hidden
        h_weights = n_hidden * n_classes
        h_bias = n_classes
        n_params = i_weights + i_bias + h_weights + h_bias
        
        return i_weights, i_bias, h_weights, h_bias, n_params
            
    
    n_inputs = 7
    n_hidden = 20
    n_classes = 1
    i_weights , i_bias, h_weights, h_bias, n_params = cal(n_inputs, n_hidden, n_classes)
    
    def forward(params):
        w1 = params[:i_weights].reshape((n_inputs, n_hidden))
        b1 = params[i_weights:i_weights+i_bias].reshape((n_hidden,))
        w2 = params[i_weights+i_bias:i_weights+i_bias+h_weights].reshape((n_hidden, n_classes))
        b2 = params[i_weights+i_bias+h_weights:].reshape((n_classes,))
        
        z1 = x_train.dot(w1) + b1
        a1 = np.where(z1 > 0, z1, z1*0.01)
        z2 = a1.dot(w2) + b2
        
        loss = mean_squared_error(y_train, z2)
        
        return loss
    
    def f(x):
        n_particles = x.shape[0]
        j = [ forward(x[i]) for i in range(n_particles)]
        return np.array(j)
    
    def train(options):
        optimizer = ps.single.GlobalBestPSO(n_particles = 100, dimensions = n_params, options = options)
        cost,pos = optimizer.optimize(f, iters = 50)
        return cost, pos, optimizer.cost_history
    
    def predict(X, pos):
        w1 = pos[:i_weights].reshape((n_inputs, n_hidden))
        b1 = pos[i_weights:i_weights+i_bias].reshape((n_hidden,))
        w2 = pos[i_weights+i_bias:i_weights+i_bias+h_weights].reshape((n_hidden, n_classes))
        b2 = pos[i_weights+i_bias+h_weights:].reshape((n_classes,))
        
        z1 = X.dot(w1) + b1
        a1 = np.where(z1 > 0, z1 , z1*0.01)
        z2 = a1.dot(w2)
        
        ypred = z2
        return ypred
    
    
    
    
    checkpoint_state = np.random.get_state()
    np.random.set_state(checkpoint_state)
    options = {'c1':0.9,'c2':0.1,'w': 0.9}
    cost , pos, history = train(options)
    
    
    y_pred = predict(x_test, pos)
    y_pred= scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(y_test)
    print(y_pred)
    status=[]
    for i in range(len(y_pred)):
        if(y_pred[i]>0.75):
            status.append("Battery in good condition")
        else:
            status.append("battery needs to be replaced")
    print("status array : ",status)
    return render_template('nnsoh.html',batterystatus=status,soh=y_test,data=x_data_orig,pred=y_pred,length=len(y_pred))
#////////////////////////////////////////////////////////////////////////////////////////////////////////

@app.route('/gasoc')
def gasoc():
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    import pickle
    from pandas import read_csv
    from keras.models import load_model

    dataset = pd.read_csv('soc1.csv', header=0, index_col=0)
    values = dataset.values
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    y = values[:, -1]
    x = values[:,:-1]
    x = scaler.fit_transform(x)
    y = scaler.fit_transform(np.array(y).reshape(-1,1))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, shuffle = False)
    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
    modelGA=load_model('modelGA.h5')
    predictY = modelGA.predict(x_test)
    
    predictY = scaler.inverse_transform(predictY)
    df = read_csv('soc1.csv')
    X_data=df.drop(['Relative State of Charge'],axis=1)
    x_data_orig=X_data[1202:]
    x_data_orig=x_data_orig.values.tolist()
    y_test = scaler.inverse_transform(y_test)
    status=[]
    for i in range(len(predictY)):
        if(predictY[i]>25):
            status.append("Battery in good condition")
        else:
           status.append("battery needs to be charged")
    print("status array : ",status)
    return render_template('gasoc.html',batterystatus=status,soh=y_test,data=x_data_orig,pred=predictY,length=len(predictY))

@app.route('/gasoh')
def gasoh():
    
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from pandas import read_csv
    from keras.models import load_model
    dataset = pd.read_csv('soh_1.csv', header=0, index_col=0)
    values = dataset.values
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    y = values[:, -1]
    x = values[:,:-1]
    x = scaler.fit_transform(x)
    y = scaler.fit_transform(np.array(y).reshape(-1,1))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, shuffle = False)
    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
    modelGAsoh=load_model('modelGASOH.h5')
    y_test = scaler.inverse_transform(y_test)
    predictY = modelGAsoh.predict(x_test)
    predictY = scaler.inverse_transform(predictY)
    df = read_csv('soh_1.csv', header=0)
    
    X_data=df.drop(['soh'],axis=1)
    x_data_orig=X_data.values.tolist()
    status=[]
    for i in range(len(predictY)):
        if(predictY[i]>0.70):
            status.append("Battery in good condition")
        else:
            status.append("battery needs to be replaced")
    print("status array : ",status)
    return render_template('gasoh.html',batterystatus=status,soh=y_test,data=x_data_orig,pred=predictY,length=len(predictY))
   
@app.route('/l2soc')
def l2soc():
    from sklearn.linear_model import Ridge
   
    df = pd.read_csv("soc.csv")
    
    X_data=df.drop(['Relative State of Charge'],axis=1)
    y_data=df['Relative State of Charge']
        
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X_data, y_data, test_size=0.2,random_state=50)
    X_testwithTimeStamp1=X_test1
    x_test_origSOC1=X_testwithTimeStamp1.values.tolist()
    X_train1=X_train1.drop(['TimeStamp'],axis=1)
    X_test1=X_test1.drop(['TimeStamp'],axis=1)
    
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    X_train1 = sc_x.fit_transform(X_train1)
    X_test1 = sc_x.transform(X_test1)
    y_train1 = sc_y.fit_transform(np.array(y_train1).reshape(-1,1))
    y_test1 = sc_y.transform(np.array(y_test1).reshape(-1,1))
    ridge_reg=Ridge(alpha=50,max_iter=100,tol=0.1)
    ridge_reg.fit(X_train1,y_train1)  
    y_prediction1 =  ridge_reg.predict(X_test1)
    actual_soc = sc_y.inverse_transform(y_test1)
    y_prediction1 = sc_y.inverse_transform(y_prediction1)

    sortto=x_test_origSOC1
    for i in range(len(y_prediction1)):
        sortto[i].append(actual_soc[i][0])
        sortto[i].append(y_prediction1[i])
    
    #for i in range(len(sortto)):
     #       sortto[i][0]=datetime.strftime(datetime.strptime(sortto[i][0],'%Y-%m-%d %H:%M:%S.%f'),'%Y-%m-%d %H:%M:%S.%f')
            
    from operator import itemgetter
    outputlist=sorted(sortto,key=itemgetter(0))   
    del outputlist[0:21]
    status=[]
    for i in range(len(outputlist)):
        if(outputlist[i][8][0]>25):
            status.append("Battery in good condition")
        else:
            status.append("battery needs to be charged")
    print("status array : ",status)
            
    return render_template('l2soc.html',batterystatus=status,
                           sortto=outputlist,length=len(outputlist))
    
@app.route('/l2soh')
def l2soh():
    df = pd.read_csv("soh_1.csv")    
    #df.describe()
    X_data=df.drop(['soh'],axis=1)
    y_data=df['soh']
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X_data, y_data, test_size=0.2,random_state=1)
    x_test_orig=X_test1
  
    x_test_orig=x_test_orig.values.tolist()
    
    X_train1=X_train1.drop(['TimeStamp'],axis=1)
    X_test1=X_test1.drop(['TimeStamp'],axis=1)
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    X_train1 = sc_x.fit_transform(X_train1)
    X_test1 = sc_x.transform(X_test1)
    y_train1 = sc_y.fit_transform(np.array(y_train1).reshape(-1,1))
    y_test1 = sc_y.transform(np.array(y_test1).reshape(-1,1))
    from sklearn.linear_model import Ridge
    ridge_reg=Ridge(alpha=50,max_iter=100,tol=0.1)
    ridge_reg.fit(X_train1,y_train1) 
    y_prediction1 =  ridge_reg.predict(X_test1)
    
    
    y_prediction2 = sc_y.inverse_transform(np.array(y_prediction1).reshape(-1,1))

    actual_soh = sc_y.inverse_transform(np.array(y_test1).reshape(-1,1))

    
    sortto=x_test_orig

    for i in range(len(y_prediction2)):
        sortto[i].append(actual_soh[i][0])
        sortto[i].append(y_prediction2[i])
    
    #for i in range(len(sortto)):
     #       sortto[i][0]=datetime.strftime(datetime.strptime(sortto[i][0],'%Y-%m-%d %H:%M:%S.%f'),'%Y-%m-%d %H:%M:%S.%f')
            
    from operator import itemgetter
    outputlist=sorted(sortto,key=itemgetter(0))
    status=[]
    for i in range(len(outputlist)):
        if(outputlist[i][9][0]>0.75):
            status.append("Battery in good condition")
        else:
            status.append("battery needs to be replaced")
    print("status array : ",status)
            
            
    #return render_template('home.html',soh=actual_soh,data=x_test_orig,pred=y_prediction2,length=len(y_prediction2))
    return render_template('l2soh.html',batterystatus=status,sortto=outputlist,length=len(sortto))










if __name__ == '__main__':
    app.run()
       
