import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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

df= pd.concat([train, test], ignore_index=True)

#srednia wartosc
avg_delay = df['Arrival Delay in Minutes'].mean()

#wypelnienie pustych rekordow srednia wartoscia 
train["Arrival Delay in Minutes"].fillna(avg_delay, inplace=True) 
test["Arrival Delay in Minutes"].fillna(avg_delay, inplace=True)

#usuniecie konkretnych kolumn
test.drop(labels = ["Unnamed: 0","id",  ], axis = 1, inplace=True)
train.drop(labels = ["Unnamed: 0","id", ], axis = 1,inplace=True)

#kodowanie tekstu na liczby unikalne całkowite
label_encoder_gender = LabelEncoder()
train['Gender'] = label_encoder_gender.fit_transform(train['Gender'])
test['Gender'] = label_encoder_gender.transform(test['Gender'])

label_encoder_class = LabelEncoder()
train['Class'] = label_encoder_class.fit_transform(train['Class'])
test['Class'] = label_encoder_class.transform(test['Class'])

label_encoder_customer_type = LabelEncoder()
train['Customer Type'] = label_encoder_customer_type.fit_transform(train['Customer Type'])
test['Customer Type'] = label_encoder_customer_type.transform(test['Customer Type'])

label_encoder_travel_type = LabelEncoder()
train['Type of Travel'] = label_encoder_travel_type.fit_transform(train['Type of Travel'])
test['Type of Travel'] = label_encoder_travel_type.transform(test['Type of Travel'])

#kodowanie float na liczby całkowite
test['Arrival Delay in Minutes'] = test['Arrival Delay in Minutes'].astype(int)
train['Arrival Delay in Minutes'] = train['Arrival Delay in Minutes'].astype(int)

#zamiana wartosci na 0 i 1
label_encoder_satisfaction = LabelEncoder()
train['satisfaction'] = label_encoder_satisfaction.fit_transform(train['satisfaction'])
test['satisfaction'] = label_encoder_satisfaction.transform(test['satisfaction'])

print(train.head())
# print(test.head())
# print(train.head())

#usuniecie kolumny sat
X_train = train.drop(['satisfaction'], axis=1)
X_test = test.drop(['satisfaction'], axis=1)

#docelowy 
y_train = train['satisfaction']

# standaryzacja danych, spojnosc
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = keras.models.Sequential()
# warstwy geste
model.add(keras.layers.Input(shape=(X_train.shape[1],)))
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))


#kompilacja modelu, optymalizator Adam
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
#trenowanie modelu
model.fit(X_train , y_train, epochs=30)

#zapisywanie modelu
model.save("my_model.keras")
loaded_model = keras.models.load_model("my_model.keras")
