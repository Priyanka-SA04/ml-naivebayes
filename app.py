from flask import Flask, request, render_template, redirect, url_for
import joblib
import numpy as np
import fitz  # PyMuPDF
import os
import re
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Load the trained model
model = joblib.load('model_nb.pkl')
category_dict = {0: 'Remember', 1: 'Understand', 2: 'Apply', 3: 'Analyse', 4: 'Create ', 5: 'Evaluate'}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function to extract text from PDF and split into questions based on numbering
def extract_questions_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text.append(page.get_text("text"))
    
    # Join all text and split based on common numbering patterns
    full_text = "\n".join(text)
    questions = re.split(r'\n\d+\.\s', full_text)
    
    # Remove any empty strings that may have been created during the split
    questions = [q.strip() for q in questions if q.strip()]
    
    return questions

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # If file is uploaded
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            questions = extract_questions_from_pdf(filepath)
            print("Extracted Questions:", questions)  # Debugging: Print extracted questions
            
            if not questions:
                return "No questions extracted from the PDF."

            predictions = model.predict(questions)
            print("Predictions:", predictions)  # Debugging: Print predictions

            results = [(question, category_dict[prediction]) for question, prediction in zip(questions, predictions)]
            os.remove(filepath)  # Clean up uploaded file after processing

            return render_template('results.html', results=results)
    
    # If individual question is entered
    if 'question' in request.form:
        question = request.form['question']
        if question.strip() == '':
            return redirect(request.url)

        prediction = model.predict([question])[0]
        result = [(question, category_dict[prediction])]
        return render_template('results.html', results=result)

    return redirect(request.url)

if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
