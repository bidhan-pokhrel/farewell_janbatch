import pandas as pd
data=pd.read_csv("classificationDataset.csv") # , sep=";)
datanonnumeric=data.drop(columns=["Age"])
dataNNcategorical= datanonnumeric.astype("category")
dataNE=datanonnumeric.apply(lambda datanonnumeric: datanonnumeric.astype("category").cat.codes)
dataNE["Age"]=data["Age"]
X= dataNE.drop(columns="Recurred")
Y=dataNE["Recurred"]
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
Xscaled=scaler.fit_transform(X)
from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest=train_test_split(Xscaled,Y,test_size=0.20,random_state=40)
import sklearn.svm
from sklearn.svm import SVC
svc_model= SVC(C=1.0, kernel="rbf", gamma="scale", random_state=40)
svc_model.fit(Xtrain,Ytrain)
svcpredicttrain=svc_model.predict(Xtrain)
svcpredicttest=svc_model.predict(Xtest)
from sklearn.metrics import accuracy_score, classification_report
print("Accuracy train: ", accuracy_score(Ytrain,svcpredicttrain))
print("Accuracy test", accuracy_score(Ytest,svcpredicttest))
import joblib
joblib.dump(svc_model, "rf_model.pkl")
print("Model saved to writing_score_model.pkl")
















