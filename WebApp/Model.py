import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


def hash_func(obj: object):
    return


@st.cache_data
def get_dataset(URL: str) -> pd.DataFrame:

    dataset = pd.read_csv(URL)

    return dataset


@st.cache_data
def fix_problems(Dataset: pd.DataFrame) -> pd.DataFrame:

    Dataset["model"] = Dataset["model"].str[1:]

    Dataset.drop(index=[32875, 35888, 16899, 87943, 52951], inplace=True)

    Dataset.reset_index(inplace=True)

    Temp_Dataset = Dataset.copy()
    Dataset = pd.DataFrame()
    for i in ["Manufacturer",
              "model",
              "transmission",
              "fuelType",
              "year",
              "engineSize",
              "mileage",
              "mpg", "tax",
              "price"]:
        Dataset[i] = Temp_Dataset[[i]]
    del Temp_Dataset

    return Dataset


@st.cache_data
def form_XCAT(Dataset: pd.DataFrame) -> pd.DataFrame:

    X_cat = Dataset.loc[:, ["Manufacturer",
                            "model", "transmission", "fuelType"]]

    return X_cat


@st.cache_data
def form_XNUM(Dataset: pd.DataFrame) -> pd.DataFrame:

    X_num = Dataset.loc[:, ["year", "engineSize", "mileage", "mpg", "tax"]]

    return X_num


@st.cache_data
def form_y(Dataset: pd.DataFrame) -> np.ndarray:

    y = Dataset.loc[:, ["price"]].values

    return y


@st.cache_resource
def fit_Encoder(X_categorical: pd.DataFrame) -> OneHotEncoder:

    Encoder = OneHotEncoder(sparse_output=False)

    Encoder.fit(X_categorical)

    return Encoder


@st.cache_data
def encode_XCAT(_Encoder: OneHotEncoder, X_categorical: pd.DataFrame) -> pd.DataFrame:

    X_categorical = pd.DataFrame(_Encoder.transform(X_categorical))

    return X_categorical


@st.cache_data
def form_X(X_categorical: pd.DataFrame, X_numerical: pd.DataFrame) -> np.ndarray:

    X = X_categorical.join(X_numerical).to_numpy()

    return X


@st.cache_data
def split_X_to_train_test(X: np.ndarray, y: np.ndarray) -> list:

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.15,
                                                        random_state=0)

    return [X_train, X_test, y_train, y_test]


@st.cache_resource
def fit_Scaler(X_train: np.ndarray) -> StandardScaler:

    Scaler = StandardScaler()

    Scaler.fit(X_train)

    return Scaler


@st.cache_data
def scale_XTRAIN(_Scaler: StandardScaler, X_train: np.ndarray) -> np.ndarray:

    X_train = _Scaler.transform(X_train)

    return X_train


@st.cache_resource
def fit_Model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:

    Model = RandomForestRegressor()

    Model.fit(X_train, y_train)

    return Model


@st.cache_data
def predict_XTEST(_Model: RandomForestRegressor, _Scaler: StandardScaler, X_test: np.ndarray) -> np.ndarray:

    y_pred = _Model.predict(_Scaler.transform(X_test))

    return y_pred


@st.cache_data
def evaluate_model(y_test: np.ndarray, y_pred: np.ndarray):

    r2 = r2_score(y_test, y_pred).round(3) * 100

    return r2
