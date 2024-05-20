#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ## Cleaning data

# In[3]:


data = pd.read_csv("subset_data.csv")
data = data[data['Status'] != 'c']


# In[4]:


data


# ## Encoding data

# In[40]:


label_encoder = LabelEncoder()
data['Sex'] = label_encoder.fit_transform(data['Sex'])


# ## Splitting data

# In[41]:


X = data[['Sex', 'Age']].values
y_days = data['Days'].values  # Days from start until removal
data['Status'].replace({'r': 0, 'd': 1}, inplace=True)  # Convert status to binary (0: dead, 1: recovered)
y_status = data['Status']
y_status_tensor = tf.convert_to_tensor(y_status, dtype=tf.float32)


# ## Standardize features

# In[42]:


scaler = StandardScaler()
X = scaler.fit_transform(X)


# ## Splitting test/train data

# In[43]:


X_train, X_test, y_days_train, y_days_test, y_status_train, y_status_test = train_test_split(X, y_days, y_status, test_size=0.2, random_state=42)


# ## Model

# In[44]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Reshape((X_train.shape[1], 1)),
    tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding="causal", activation="relu"),
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.LSTM(32), 
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1, name='days_output'),
    tf.keras.layers.Lambda(lambda x: x * 400)
])
# Retrieve output of the last layer in the Sequential model
sequential_output = model.layers[-1].output

# Additional output layer for status prediction
status_output = tf.keras.layers.Dense(1, activation='sigmoid', name='status_output')(sequential_output)

model = tf.keras.models.Model(inputs=model.inputs, outputs=[model.layers[-1].output, status_output])


# ## Compiling Model

# In[45]:


model.compile(loss=[tf.keras.losses.Huber(), 'binary_crossentropy'],
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              metrics=["mae", "accuracy"])


# ## Train Model

# In[46]:



model.fit(X_train, [y_days_train, y_status_train], epochs=25, batch_size=5444, verbose=1)


# ## Evaluate the model

# In[48]:


test_loss, test_mae_days, test_mae_status = model.evaluate(X_test, [y_days_test, y_status_test], verbose=0)
print('Test MAE (Mean Absolute Error) - Days:', test_mae_days)
print('Test MAE (Mean Absolute Error) - Status:', test_mae_status)


# ## Prediction

# In[50]:


y_days_pred, y_status_pred = model.predict(X_test)


# ## Accuracy check for status

# In[51]:


y_status_pred_classes = np.argmax(y_status_pred, axis=1)
y_status_pred_classes


# In[33]:



accuracy = accuracy_score(y_status_test, y_status_pred_classes)
precision = precision_score(y_status_test, y_status_pred_classes, average='weighted')
recall = recall_score(y_status_test, y_status_pred_classes, average='weighted')
f1 = f1_score(y_status_test, y_status_pred_classes, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# ## Visualization

# In[52]:


plt.figure(figsize=(10, 6))
plt.hist(y_status_pred_classes, bins=20, color='skyblue', edgecolor='black', label='Predicted')
plt.hist(y_status_test, bins=20, color='green', edgecolor='black', alpha= 0.5,label='Actual')
plt.title('Predicted Status Probabilities')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()


# In[53]:



indices = np.arange(len(y_days_test))

sorted_indices = np.argsort(y_days_test)

plt.figure(figsize=(10, 6))
plt.plot(indices, y_days_test[sorted_indices], label='Actual', color='blue', alpha = 0.5)
plt.plot(indices, y_days_pred[sorted_indices], label='Predicted', color='red', alpha = 0.5)
plt.title('Predicted vs. Actual Days until Removal')
plt.xlabel('Instance')
plt.ylabel('Days until Removal')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




