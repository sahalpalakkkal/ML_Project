import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100)

# Create your views here.


def pred(request):

    return render(request, 'prediction.html')


def result(request):
    data = pd.read_csv(
        r"C:\Users\RKD\Downloads\sahal project files\Vehicle Insurance data new[1].csv")
    x = data.drop(["premium amount"], axis=1)
    label_encoder = LabelEncoder()
    x['Vehicle Name'] = label_encoder.fit_transform(x['Vehicle Name'])
    x['Vehicle Damage'] = label_encoder.fit_transform(x['Vehicle Damage'])
    x['Insurance Type'] = label_encoder.fit_transform(x['Insurance Type'])
    y = data['premium amount']
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.12)
    model = RandomForestRegressor()
    model.fit(xtrain, ytrain)
    model.score(xtest, ytest)

    joblib.dump(model, 'joblib_model')
    new_model = joblib.load('joblib_model')

    v1 = request.GET['Vehicle name']
    v2 = request.GET['Vehicle age']
    v3 = request.GET['Vehicle damage']
    v4 = request.GET['Insurance type']
    v5 = request.GET['Duration']

    print("v1",v1)
    print("v2",v2)
    print("v3",v3)
    print("v4",v4)
    print("v5",v5)
    print()

    x_val=[v1,v3,v4]
    x_val=label_encoder.fit_transform(x_val)
    x_val=np.insert(x_val,1,v2)
    x_val=np.append(x_val,v5)
    pred = model.predict([x_val])
    print("x_val",x_val)
    print("pred",pred)
    result1 =int(pred)

    return render(request, 'prediction.html', {'result2': result1})
