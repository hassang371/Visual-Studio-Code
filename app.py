from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import subprocess
import os
import sys
import uuid

# App setup
app = Flask(__name__)

# Ensure necessary directories exist
os.makedirs(os.path.join(os.getcwd(), 'uploads'), exist_ok=True)
os.makedirs(os.path.join(os.getcwd(), 'static', 'visualizations'), exist_ok=True)
print("Uploads and visualizations directories ensured.")

# Handle CORS preflight requests globally
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

# Handle CORS preflight for /train
@app.route('/train', methods=['OPTIONS'])
def train_options():
    response = jsonify({'message': 'CORS preflight successful'})
    return add_cors_headers(response)

# Improved Training Route with Detailed Error Handling
@app.route('/train', methods=['POST'])
def train():
    try:
        health_file = request.files.get('healthFile')
        wearable_files = request.files.getlist('wearableFiles')

        if not health_file or not wearable_files:
            response = jsonify({'error': 'Health data and wearable data files are required'})
            return add_cors_headers(response), 400

        print(f"Received health file: {health_file.filename}")
        print(f"Received {len(wearable_files)} wearable files.")

        # Create unique directory for this training session
        session_id = str(uuid.uuid4())
        session_dir = os.path.join(os.getcwd(), 'uploads', session_id)
        os.makedirs(session_dir, exist_ok=True)

        # Save uploaded health file
        health_path = os.path.join(session_dir, 'healthcare_dataset.csv')
        health_file.save(health_path)

        # Save uploaded wearable files
        wearable_paths = []
        for i, file in enumerate(wearable_files):
            wearable_path = os.path.join(session_dir, f'wearable_data_{i}.csv')
            file.save(wearable_path)
            wearable_paths.append(wearable_path)

        # Run the training script, passing the file paths as arguments
        command = [sys.executable, 'new.py', health_path] + wearable_paths
        print("Starting model training...")
        result = subprocess.run(command, capture_output=True, text=True)

        print("Training script output:", result.stdout)
        print("Training script error:", result.stderr)

        if result.returncode != 0:
            error_message = result.stderr if result.stderr else 'Unknown error occurred.'
            print("Training failed with error:", error_message)
            response = jsonify({'error': 'Training failed', 'details': error_message})
            return add_cors_headers(response), 500

        print("Model training completed successfully.")

        # Get visualizations
        visualization_dir = os.path.join(os.getcwd(), 'static', 'visualizations')
        os.makedirs(visualization_dir, exist_ok=True)
        visualizations = [f for f in os.listdir(visualization_dir) if f.endswith('.png')]

        if not visualizations:
            print("Warning: No visualizations generated.")
            response = jsonify({'message': 'Model trained, but no visualizations were generated.'})
        else:
            response = jsonify({'message': 'Model trained successfully', 'visualizations': visualizations})

        return add_cors_headers(response), 200

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        response = jsonify({'error': 'An unexpected error occurred', 'details': str(e)})
        return add_cors_headers(response), 500

# Visualization Route
@app.route('/visualizations/<path:filename>')
def get_visualization(filename):
    visualization_dir = os.path.join(os.getcwd(), 'static', 'visualizations')
    return send_from_directory(visualization_dir, filename)

if __name__ == '__main__':
    app.run(debug=True)