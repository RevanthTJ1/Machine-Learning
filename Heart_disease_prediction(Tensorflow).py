#Important note: Tensorflow does not support Python 3.7 (At the time of uploading the source code). The same version is supported on Ubuntu. For Windows Till Tensorflow is supported till Python 3.6
#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#Importing the dataset
df=pd.read_csv("heart.csv")

#choosing the columns for normalization
cols_to_norm = ['sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang', 'oldpeak', 'slope', 'ca', 'thal']

#Min-Max Scaler (Bringing the columns between 0 to 1 range to avoid biasing)
df[cols_to_norm] = df[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

# creating tensorflow feature columns
Age = tf.feature_column.numeric_column('age')
gender = tf.feature_column.numeric_column('sex')
Cp = tf.feature_column.numeric_column('cp')
trest_bps = tf.feature_column.numeric_column('trestbps')
Chol = tf.feature_column.numeric_column('chol')
Fbs = tf.feature_column.numeric_column('fbs')
rest_ecg = tf.feature_column.numeric_column('restecg')
thal_ach = tf.feature_column.numeric_column('thalach')
ex_ang = tf.feature_column.numeric_column('exang')
old_peak = tf.feature_column.numeric_column('oldpeak')
Slope = tf.feature_column.numeric_column('slope')
Ca = tf.feature_column.numeric_column('ca')
Thal = tf.feature_column.numeric_column('thal')

# forming age groups
age_groups = tf.feature_column.bucketized_column(Age, boundaries=[20,30,40,50,60,70,80])

feat_cols = [age_groups, gender , Cp , trest_bps , Chol , Fbs , rest_ecg , thal_ach , ex_ang , old_peak , Slope ,Ca, Thal]

#feature selection
x_data = df.iloc[:,:13]
labels = df.iloc[:,13:14]

#Training and testing the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_data, labels, test_size=0.3, random_state=42)

#Building the network (Input and hidden nodes)
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=500,shuffle=True)
model = tf.estimator.DNNClassifier(hidden_units=[14,14,14] , feature_columns=feat_cols)

model.train(input_fn=input_func,steps = 1000)

# input function for evaluation
eval_input_func = tf.estimator.inputs.pandas_input_fn( x=X_test , y=y_test , batch_size=10 , num_epochs=1 , shuffle=False )

results = model.evaluate(eval_input_func)

# input function for prediction
pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=10,num_epochs=1,shuffle=False)

# store all predictions
predictions = list(model.predict(pred_input_func))

# obtain class_ids which stores the prediction value of 0 or 1
final_preds = []

for pred in predictions:
    final_preds.append(pred['class_ids'][0])
 
#Model Evaluation    
from sklearn.metrics import classification_report
print(classification_report(y_test,final_preds))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,final_preds)
print(cm)

plt.matshow(cm)
plt.title('Confusion matrix of the classifier\n')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.colorbar()
plt.show()