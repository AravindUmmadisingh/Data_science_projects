import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('Sms.csv', encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'message']
data['label'] = data['label'].map({'spam': 1, 'ham': 0})

# Split the data
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

# Convert text to features
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluate the model
y_pred = model.predict(X_test_vec)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Predict on new data
new_messages = ["Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.",
                "Hey, are we still meeting for lunch today?"]
new_messages_vec = vectorizer.transform(new_messages)
predictions = model.predict(new_messages_vec)
for msg, pred in zip(new_messages, predictions):
    label = 'spam' if pred == 1 else 'ham'
    print(f'Message: {msg} \nPredicted: {label}\n')
