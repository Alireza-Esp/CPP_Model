import sys
sys.path.append("..")


import streamlit as st
import Model
from pathlib import Path

Dataset_URL = Path.cwd().parent.absolute(
).parent.absolute().joinpath("Data\CarsData.csv")

Dataset = Model.get_dataset(Dataset_URL)

Dataset = Model.fix_problems(Dataset)

X_cat = Model.form_XCAT(Dataset)
X_num = Model.form_XNUM(Dataset)
y = Model.form_y(Dataset)

Encoder = Model.fit_Encoder(X_cat)

X_cat = Model.encode_XCAT(Encoder, X_cat)

X = Model.form_X(X_cat, X_num)

X_train, X_test, y_train, y_test = Model.split_X_y_to_train_test(X, y)

Scaler = Model.fit_Scaler(X_train)

X_train = Model.scale_XTRAIN(Scaler, X_train)

CPP_Model = Model.fit_Model(X_train, y_train)

y_pred = Model.predict_XTEST(CPP_Model, Scaler, X_test)

score = Model.evaluate_model(y_test, y_pred)
