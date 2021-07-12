import pandas as pd
import streamlit as st
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
df = pd.read_csv("iris-species.csv")
df["Label"] = df["Species"].map({"Iris-setosa" : 0, "Iris-virginica" : 1, "Iris-versicolor" : 2})
x = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = df["Label"]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 42, test_size = 0.3)
svc_model = SVC(kernel = "linear")
svc_model.fit(x_train, y_train)
rfc = RandomForestClassifier(n_jobs = -1, n_estimators = 100).fit(x_train, y_train)
lg = LogisticRegression().fit(x_train, y_train)
dict_1 = {"Support Vector Machine" : "svc", "Logistic Regression" : "log", "Random Forest Classifier" : "rf"}
classifier = st.sidebar.selectbox('Classifier', ('Support Vector Machine', 'Logistic Regression', 'Random Forest Classifier'))
@st.cache()
def prediction(sl, sw, pl, pw, model):
  if model == "svc":
    pred = svc_model.predict([[sl, sw, pl, pw]])
  elif model == "log":
    pred = lg.predict([[sl, sw, pl, pw]])
  else:
    pred = rfc.predict([[sl, sw, pl, pw]])
  pred = pred[0]
  if pred == 0:
    return "Iris-setosa"
  elif pred == 1:
    return "Iris-virginica"
  else:
    return "Iris-versicolor"
st.sidebar.title("IRIS SPECIES CLASSIFIER")
sl = st.sidebar.slider("SepalLength:-", float(df["SepalLengthCm"].min()), float(df["SepalLengthCm"].max()))
sw = st.sidebar.slider("SepalWidth:-", float(df["SepalWidthCm"].min()), float(df["SepalWidthCm"].max()))
pl = st.sidebar.slider("PetalLength:-", float(df["PetalLengthCm"].min()), float(df["PetalLengthCm"].max()))
pw = st.sidebar.slider("PetalWidth:-", float(df["PetalWidthCm"].min()), float(df["PetalWidthCm"].max()))
if st.sidebar.button("PREDICT"):
  model = dict_1[classifier]
  if model == "svc":
    accuracy = svc_model.score(x_train, y_train)
  elif model == "log":
    accuracy = lg.score(x_train, y_train)
  else:
    accuracy = rfc.score(x_train, y_train)
  accuracy = str(round(accuracy * 100, 3)) + "%"
  predicted = prediction(sl, sw, pl, pw, model)
  st.write("The species is ", predicted)
  st.write("The accuracy of the model is ", accuracy)