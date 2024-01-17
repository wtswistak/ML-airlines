import numpy as np
import pandas as pd
import time
import pickle
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
print("Tensorflow version: ", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train["Arrival Delay in Minutes"].fillna(0, inplace=True)
test["Arrival Delay in Minutes"].fillna(0, inplace=True)

label_encoder_gender = LabelEncoder()
train['Gender'] = label_encoder_gender.fit_transform(train['Gender'])
test['Gender'] = label_encoder_gender.transform(test['Gender'])

label_encoder_customer_type = LabelEncoder()
train['Customer Type'] = label_encoder_customer_type.fit_transform(train['Customer Type'])
test['Customer Type'] = label_encoder_customer_type.transform(test['Customer Type'])

label_encoder_travel_type = LabelEncoder()
train['Type of Travel'] = label_encoder_travel_type.fit_transform(train['Type of Travel'])
test['Type of Travel'] = label_encoder_travel_type.transform(test['Type of Travel'])

label_encoder_class = LabelEncoder()
train['Class'] = label_encoder_class.fit_transform(train['Class'])
test['Class'] = label_encoder_class.transform(test['Class'])

X_train = train.drop(['id', 'satisfaction'], axis=1)
y_train = train['satisfaction']

X_test = test.drop(['id'], axis=1)

y_train = pd.get_dummies(y_train)

scaler = StandardScaler()
X_train[['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']] = scaler.fit_transform(X_train[['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']])
X_test[['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']] = scaler.transform(X_test[['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']])

model = keras.models.Sequential()
model.add(keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(2, activation='softmax'))

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, validation_split=0.2)

model.save("my_model.h5")

loaded_model = keras.models.load_model("my_model.h5")

X_test = test.drop(['id', 'satisfaction'], axis=1)
X_test[['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']] = scaler.transform(X_test[['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']])

predictions = loaded_model.predict(X_test)



