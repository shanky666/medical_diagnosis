import pandas as pd
from flask import Flask, request, render_template
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import numpy as np

# Load the dataset
data = pd.read_csv('dataset.csv')

# Fill NaN values with empty strings
data.fillna('', inplace=True)

# Combine all symptom columns into a single list of symptoms
data['Symptoms'] = data[['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'Symptom_5', 
                          'Symptom_6', 'Symptom_7', 'Symptom_8', 'Symptom_9', 'Symptom_10', 
                          'Symptom_11', 'Symptom_12', 'Symptom_13', 'Symptom_14', 
                          'Symptom_15', 'Symptom_16', 'Symptom_17']].values.tolist()

# Remove empty strings from symptom lists
data['Symptoms'] = data['Symptoms'].apply(lambda x: [symptom for symptom in x if symptom])

# Encode symptoms using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(data['Symptoms'])

# Encode diseases using LabelEncoder
le = LabelEncoder()
y = le.fit_transform(data['Disease'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train using Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Create the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', symptoms=mlb.classes_)

@app.route('/predict', methods=['POST'])
def predict():
    selected_symptoms = request.form.getlist('symptoms')
    symptom_vector = mlb.transform([selected_symptoms])
    
    # Predict probabilities
    probabilities = model.predict_proba(symptom_vector)[0]
    
    # Get top 3 diseases
    top_indices = np.argsort(probabilities)[-3:][::-1]
    top_diseases = le.inverse_transform(top_indices)
    top_probabilities = probabilities[top_indices]
    
    disease_probabilities = {disease: prob for disease, prob in zip(top_diseases, top_probabilities)}
    
    return render_template('result.html', diseases=disease_probabilities)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
