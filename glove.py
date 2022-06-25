import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from sklearn.utils import shuffle
from imblearn.pipeline import Pipeline
import lightgbm as lgb

df = pd.read_csv('glove_data.csv')
features = np.array(df.columns[:-1])
target = 'Target'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

def xgb_shap_values(X_train, y_train, X_test):

    model = XGBClassifier()
    model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model, X_train)
    shap_values_xgb = explainer.shap_values(X_test)
    # shap.summary_plot(shap_values_xgb, X_test)
    # shap.summary_plot(shap_values_xgb, X_test, plot_type ='bar')
    # shap.dependence_plot("Age", shap_values_xgb, X_test, interaction_index = 'rate Po2')
    shap_sum = np.abs(shap_values_xgb).mean(axis=0)
    importance_df = pd.DataFrame([features, shap_sum.tolist()]).T
    importance_df.columns = ['Column Name', 'Shap Importance']
    importance_df= importance_df.sort_values('Shap Importance', ascending = False)
    return importance_df

selected_df = xgb_shap_values(X_train, y_train,  X_test)
selected_df = selected_df[0:400]
selected_df.shape
cols = selected_df['Column Name']
## converting df to numpy array/list
trainingdf = np.array(cols)
df1 = X[trainingdf]

X = df1
y = y

count_classes = pd.value_counts(y, sort = True)

count_classes.plot(kind = 'bar', rot=0)

plt.title("Class Distribution")

LABELS = ["Zero", "One"]

plt.xticks(range(2), LABELS)

plt.xlabel("Class")

plt.ylabel("Frequency")

One = df[y==1]

Zero = df[y==0]

print(Zero.shape,One.shape)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 50)

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPool1D

from tensorflow.keras.optimizers import Adam

import  seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = X_train.reshape(1683, 400,1)
X_test = X_test.reshape(421, 400, 1)

epochs = 50
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=4, activation='relu', input_shape = (400,1))) 
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Conv1D(filters=64, kernel_size=4, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Flatten())   
model.add(Dense(32, activation='relu')) 
model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid')) 

model.compile(optimizer=Adam(learning_rate=0.0002), loss = 'binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=epochs, batch_size=64,validation_data=(X_test, y_test), verbose=1)

from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = model.predict(X_test) 
y_pred = (y_pred > 0.5) 

cmann = confusion_matrix(y_test,y_pred) 
print(cmann)

accuracy = (cmann[0][0]+cmann[1][1])/(cmann[0][1] + cmann[1][0] +cmann[0][0] +cmann[1][1]) 
print(accuracy*100)

p = model.predict(X_test)
from sklearn.metrics import roc_auc_score
print("AUC = ",roc_auc_score(y_test,p))

from sklearn.metrics import matthews_corrcoef,confusion_matrix
matthews_corrcoef(y_test,y_pred)

TP = cmann[0][0]
TN = cmann[1][1]
FP = cmann[0][1]
FN = cmann[1][0]
print('True Positives:', TP)
print('True Negatives:', TN)
print('False Positives:', FP)
print('False Negatives:', FN)

sensitivity=(TP/(TP+FN))
sensitivity

specificity = (TN/(TN+FP))
specificity

precision = (TP/(TP+FP))
recall = (TP/(TP+FN))
print(precision)
print(recall)

f1 = (2*(precision*recall))/(precision+recall)
f1
