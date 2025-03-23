from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin
import subprocess
import os
import sys
import uuid

# App setup
app = Flask(__name__)
CORS(app, resources={r"/train": {"origins": "http://127.0.0.1:8000"}})

# Ensure necessary directories exist
os.makedirs(os.path.join(os.getcwd(), 'uploads'), exist_ok=True)
os.makedirs(os.path.join('static', 'visualizations'), exist_ok=True)
print("Uploads and visualizations directories ensured.")

# Handle CORS preflight for /train
@app.route('/train', methods=['OPTIONS'])
def train_options():
    response = jsonify({'message': 'CORS preflight successful'})
    return response

# Improved Training Route with Detailed Error Handling
@app.route('/train', methods=['POST'])
@cross_origin()
def train():
    try:
        health_file = request.files.get('healthData')
        activity_file = request.files.get('activityData')
        digital_interaction_file = request.files.get('digitalInteractionData')
        personal_health_file = request.files.get('personalHealthData')

        if not health_file or not activity_file or not digital_interaction_file or not personal_health_file:
            response = jsonify({'error': 'All four datasets (health, activity, digital interaction, personal health) are required'})
            return response, 400

        print(f"Received health file: {health_file.filename}")
        print(f"Received activity file: {activity_file.filename}")
        print(f"Received digital interaction file: {digital_interaction_file.filename}")
        print(f"Received personal health file: {personal_health_file.filename}")

        # Create unique directory for this training session
        session_id = str(uuid.uuid4())
        session_dir = os.path.join(os.getcwd(), 'uploads', session_id)
        os.makedirs(session_dir, exist_ok=True)

        # Save uploaded files
        health_path = os.path.join(session_dir, f"healthcare_dataset_{session_id}.csv")
        activity_path = os.path.join(session_dir, f"activity_environment_data_{session_id}.csv")
        digital_interaction_path = os.path.join(session_dir, f"digital_interaction_data_{session_id}.csv")
        personal_health_path = os.path.join(session_dir, f"personal_health_data_{session_id}.csv")

        health_file.save(health_path)
        activity_file.save(activity_path)
        digital_interaction_file.save(digital_interaction_path)
        personal_health_file.save(personal_health_path)

        # Check if files are saved correctly
        for path, name in zip(
            [health_path, activity_path, digital_interaction_path, personal_health_path],
            ['Health', 'Activity', 'Digital Interaction', 'Personal Health']
        ):
            if not os.path.exists(path):
                print(f"Error: {name} file not saved properly.")
                response = jsonify({'error': f'{name} file could not be saved.'})
                return response, 500

        # Run the training script, passing the file paths as arguments
        command = [sys.executable, 'new.py', health_path, activity_path, digital_interaction_path, personal_health_path]
        print("Starting model training...")
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=300)
            print("Training script output:", result.stdout)
            print("Training script error (if any):", result.stderr)
        except subprocess.CalledProcessError as e:
            print("Subprocess failed:", e.stderr if e.stderr else e.stdout)
            response = jsonify({'error': 'Training script failed', 'details': e.stderr if e.stderr else e.stdout})
            return response, 500
        except subprocess.TimeoutExpired:
            print("Subprocess timed out after 300 seconds.")
            response = jsonify({'error': 'Training script timed out after 300 seconds'})
            return response, 500

        print("Model training completed without errors.")

        # Get visualizations
        visualization_dir = os.path.join('static', 'visualizations')
        os.makedirs(visualization_dir, exist_ok=True)
        visualizations = [f for f in os.listdir(visualization_dir) if f.endswith('.png')]

        if not visualizations:
            print("Warning: No visualizations generated.")
            response = jsonify({'message': 'Model trained, but no visualizations were generated.'})
        else:
            response = jsonify({'message': 'Model trained successfully', 'visualizations': visualizations})

        return response, 200

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        response = jsonify({'error': 'An unexpected error occurred', 'details': str(e)})
        return response, 500

# Visualization Route
@app.route('/visualizations/<path:filename>')
def get_visualization(filename):
    return send_from_directory('static/visualizations', filename)

if __name__ == '__main__':
    app.run(debug=True)