import numpy as np
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK datasets
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Process text by lowercasing, removing non-alphanumeric characters,
    tokenizing, removing stopwords, and lemmatizing."""
    text = re.sub(r'\W', ' ', str(text).lower())
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def load_and_preprocess_data(filepath):
    """Load data from a CSV file, preprocess text, and balance the dataset dynamically."""
    print("Loading dataset...")
    df = pd.read_csv(filepath, delimiter=';')
    print("Dataset loaded. Here are a few rows from the dataset:")
    print(df.head())  # Print the first few rows of the DataFrame to confirm data is loaded
    print("Unique labels in the dataset:")
    print(df['label'].unique())  # Print all unique labels

    print("Preprocessing text...")
    df['title'] = df['title'].apply(preprocess_text)
    print("Text preprocessing completed.")

    # Determine the smallest class size dynamically
    min_class_size = df['label'].value_counts().min()
    print(f"Balancing all classes to the smallest class size: {min_class_size}")

    df_balanced = pd.DataFrame()
    for label in df['label'].unique():
        df_subset = df[df['label'] == label]
        df_balanced = pd.concat([df_balanced, df_subset.sample(min_class_size, replace=False)])

    print("Balanced dataset sample sizes for each class:")
    print(df_balanced['label'].value_counts())
    return df_balanced

# List files in the input directory
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Load and preprocess the data
df_balanced = load_and_preprocess_data('full_dataset_v3.csv')

# Encoding the labels
encoder = LabelEncoder()
df_balanced['category'] = encoder.fit_transform(df_balanced['label'])
y = df_balanced['category']
X = df_balanced['title']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline and hyperparameter tuning
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('nb', MultinomialNB())
])
parameters = {
    'nb__alpha': (0.001, 0.01, 0.1, 0.5, 0.8, 1, 2, 5, 10),
}
grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print("Best Score: %s" % grid_search.best_score_)
print("Best Hyperparameters: %s" % grid_search.best_params_)

means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
alphas = grid_search.cv_results_['param_nb__alpha']

print("\nAlpha Mean Accuracy Confidence Interval")
for mean, std, alpha in zip(means, stds, alphas):
    print(f"{alpha} {mean:.5f} +/-{2*std:.3f}")

# Evaluating the best model
y_pred = grid_search.predict(X_test)
report = classification_report(y_test, y_pred, target_names=encoder.classes_)
print(report)
print("hello")

# Get wrongly classified examples
wrong_preds = y_test != y_pred
wrong_examples = X_test[wrong_preds]
wrong_true_labels = y_test[wrong_preds]
wrong_pred_labels = y_pred[wrong_preds]

# Print wrongly classified examples
print("\nWrongly Classified Examples:")
for text, true_label, predicted_label in zip(wrong_examples, wrong_true_labels, wrong_pred_labels):
    print(f"Title: {text}")
    print(f"True Label: {encoder.classes_[true_label]}, Predicted Label: {encoder.classes_[predicted_label]}")
    print("-----")





# Compute and display the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


