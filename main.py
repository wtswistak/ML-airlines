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



# print(train_data.isnull().sum()) #brakujace wartosic
# print(test_data.isnull().sum())

# train_data["Arrival Delay in Minutes"].fillna(0, inplace = True) #wypelnienie pustych rekordow
# test_data["Arrival Delay in Minutes"].fillna(0, inplace = True)


# X_train = train_data.drop('satisfaction', axis=1)
# y_train = train_data['satisfaction']



# X_train_split, X_val, y_train_split, y_val = train_test_split(X_train_normalized, y_train, test_size=0.2, random_state=42)

# model=keras.models.Sequential()

# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentr', metrics=['accuracy'])

# history = model.fit(X_train_split, y_train_split, epochs=10, validation_data=(X_val, y_val))

# train = pd.read_csv('train.csv')
# test = pd.read_csv('test.csv')

# train["Arrival Delay in Minutes"].fillna(0, inplace = True) #wypelnienie pustych rekordow
# test["Arrival Delay in Minutes"].fillna(0, inplace = True)

# label_encoder_gender = LabelEncoder()
# train['Gender'] = label_encoder_gender.fit_transform(train['Gender'])
# test['Gender'] = label_encoder_gender.transform(test['Gender'])

# # Analogicznie, użyj tego samego obiektu LabelEncoder dla 'Customer Type', 'Type of Travel' i 'Class'
# label_encoder_customer_type = LabelEncoder()
# train['Customer Type'] = label_encoder_customer_type.fit_transform(train['Customer Type'])
# test['Customer Type'] = label_encoder_customer_type.transform(test['Customer Type'])

# label_encoder_travel_type = LabelEncoder()
# train['Type of Travel'] = label_encoder_travel_type.fit_transform(train['Type of Travel'])
# test['Type of Travel'] = label_encoder_travel_type.transform(test['Type of Travel'])

# label_encoder_class = LabelEncoder()
# train['Class'] = label_encoder_class.fit_transform(train['Class'])
# test['Class'] = label_encoder_class.transform(test['Class'])

# # Podziel dane na cechy i zmienną docelową
# X_train = train.drop(['id', 'satisfaction'], axis=1)
# y_train = train['satisfaction']

# X_test = test.drop(['id'], axis=1)

# # One-hot encode zmienną docelową
# y_train = pd.get_dummies(y_train)

# # Znormalizuj cechy numeryczne
# scaler = StandardScaler()
# X_train[['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']] = scaler.fit_transform(X_train[['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']])
# X_test[['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']] = scaler.transform(X_test[['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']])

# # Zbuduj model
# model=keras.models.Sequential()

# # Skompiluj model
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentr', metrics=['accuracy'])

# # Trenuj model
# # model.fit(X_train.values.astype('float32'), y_train, epochs=10, validation_split=0.2)
# model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
# # Zapisz model
# model.save("my_model.keras")

# # Wczytaj zapisany model
# loaded_model = keras.models.load_model("my_model.keras")

# # Przygotuj dane testowe (bez kolumny 'satisfaction')
# X_test = test.drop(['id', 'satisfaction'], axis=1)
# # Znormalizuj cechy numeryczne w danych testowych
# X_test[['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']] = scaler.transform(X_test[['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']])

# # Ocen model na zbiorze testowym
# predictions = loaded_model.predict(X_test.values.astype('float32'))



# # print(train_data.isnull().sum()) #brakujace wartosic
# # print(test_data.isnull().sum())

# # train_data["Arrival Delay in Minutes"].fillna(0, inplace = True) #wypelnienie pustych rekordow
# # test_data["Arrival Delay in Minutes"].fillna(0, inplace = True)


# # X_train = train_data.drop('satisfaction', axis=1)
# # y_train = train_data['satisfaction']



# # X_train_split, X_val, y_train_split, y_val = train_test_split(X_train_normalized, y_train, test_size=0.2, random_state=42)

# # model=keras.models.Sequential()

# # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentr', metrics=['accuracy'])

# # history = model.fit(X_train_split, y_train_split, epochs=10, validation_data=(X_val, y_val))