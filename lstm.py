import numpy as np # linear algebra
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D, Bidirectional, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Dropout, Dense, Bidirectional
import seaborn as sns
import os



for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#load the file and process the data
df = pd.read_csv('full_dataset_v3.csv',sep=';')
df.head()
df.groupby('label').describe()
df_new = df[['title', 'label']]
df_r = df_new[df_new['label']=='R']
df_java = df_new[df_new['label']=='java']
df_java = df_java.sample(df_r.shape[0])
df_js = df_new[df_new['label']=='javascript']
df_js = df_js.sample(df_r.shape[0])
df_php = df_new[df_new['label']=='php']
df_php = df_php.sample(df_r.shape[0])
df_python = df_new[df_new['label']=='python']
df_python = df_python.sample(df_r.shape[0])
df_balanced = pd.concat([df_r, df_java, df_js, df_php, df_python])
df_balanced['label'].value_counts()
df_balanced['category'] = df_balanced['label'].apply(lambda x: 0 if x == 'R' else 1 if x == 'java' else 2 if x == 'javascript' else 3 if x == 'php' else 4)

X = df_balanced['title'].values
y = df_balanced['category'].values

# Create tokenized and padded sequences
tokenizer = Tokenizer(num_words=40000, oov_token="<OOV>")
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
max_len = max(len(x) for x in sequences) + 20
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

# Split the data, preserving the original indices
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
    padded_sequences, y, range(len(df_balanced)), test_size=0.2, random_state=42
)

titles = df_balanced.iloc[indices_test]['title']

lstm_units = 256
dropout_rate = 0.5
optimizer = 'adam'
# Model building and training as before
model = Sequential([
    Embedding(input_dim=40000, output_dim=64, input_length=max_len),
    SpatialDropout1D(0.2),
    Bidirectional(LSTM(lstm_units)),  # Using a Bidirectional LSTM with increased units
    Dropout(dropout_rate),  # Added a Dropout layer to reduce overfitting
    Dense(5, activation='softmax')  # Adjust the number of output units to match the number of classes
])
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=4, batch_size=32, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Predicting the test set results
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels


# Generating the confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 7))  # Adjust the figure size as needed
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Scale'})
plt.xlabel('Actual label')
plt.ylabel('Predict label')
plt.title('Confusion Matrix for LSTM')  # Adjust the title as per your model
plt.show()

# Generating classification report for precision, recall, and F1-score
cr = classification_report(y_test, y_pred_classes)
print("Classification Report:")
print(cr)

