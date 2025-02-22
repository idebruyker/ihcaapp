from flask import Flask, render_template, request, send_file
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

app = Flask(__name__)

# Load the saved model
model_filename = 'best_model.pkl'

if os.path.isfile(model_filename):
    print(f'Loading model from: {model_filename}')
else:
    print(f'Model file not found: {model_filename}')

loaded_model = joblib.load(model_filename)
# Configure upload and processed folders
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Process the CSV file
            df = pd.read_csv(filepath)

            # Make predictions on the new dataset
            predictions = loaded_model.predict(df)
            # Add predictions to the new dataset 
            df['Predictions'] = predictions

            # Example: Generate a plot
            # plt.figure()
            # df.plot(kind='bar')  # Example plot (customize as needed)
            # plot_path = os.path.join(app.config['PROCESSED_FOLDER'], 'plot.png')
            # plt.savefig(plot_path)

            # Example: Save processed CSV
            # processed_csv_path = os.path.join(app.config['PROCESSED_FOLDER'], 'processed_output.csv')
            # df.to_csv(processed_csv_path, index=False)

            # Provide download links
            return render_template('index.html', 
                                 plot_url=plot_path, 
                                 csv_url=processed_csv_path)

    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['PROCESSED_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)