import os
import json
import numpy as np
import tensorflow as tf
from waitress import serve
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageDraw 
import cv2
import io
import base64
from flask import Flask, render_template, request, jsonify, url_for, redirect, send_from_directory

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create images folder for static assets if it doesn't exist
IMAGES_FOLDER = 'static/images'
if not os.path.exists(IMAGES_FOLDER):
    os.makedirs(IMAGES_FOLDER)

# Create folder for processed images (with face detection visualization)
PROCESSED_FOLDER = 'static/processed'
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

# Save hero image SVG to the images folder
def save_hero_image():
    svg_path = os.path.join(IMAGES_FOLDER, 'hero-image.svg')
    if not os.path.exists(svg_path):
        try:
            with open('hero-image.svg', 'r') as f:
                svg_content = f.read()
            
            with open(svg_path, 'w') as f:
                f.write(svg_content)
        except:
            print("Could not save hero image SVG. Please make sure the file exists.")

# MST colors for reference
MST_COLORS = [
    '#f6ede4',  # MST1
    '#f3e7db',  # MST2
    '#f7ead0',  # MST3
    '#eadaba',  # MST4
    '#d7bd96',  # MST5
    '#a07e56',  # MST6
    '#825c43',  # MST7
    '#604134',  # MST8
    '#3a312a',  # MST9
    '#292420'   # MST10
]

# Clothing color recommendations by skin tone group
CLOTHING_RECOMMENDATIONS = {
    'light': {  # MST1-2
        'recommended': [
            {'name': 'Navy Blue', 'hex': '#000080'},
            {'name': 'Royal Purple', 'hex': '#7851a9'},
            {'name': 'Emerald Green', 'hex': '#046307'},
            {'name': 'Burgundy', 'hex': '#800020'},
            {'name': 'Sapphire Blue', 'hex': '#0f52ba'}
        ],
        'avoid': [
            {'name': 'Orange', 'hex': '#ffa500'},
            {'name': 'Bright Yellow', 'hex': '#ffff00'},
            {'name': 'Pastel Colors', 'hex': '#fadadd'}
        ]
    },
    'light medium': {  # MST3-4
        'recommended': [
            {'name': 'Teal', 'hex': '#008080'},
            {'name': 'Cobalt Blue', 'hex': '#0047ab'},
            {'name': 'Lavender', 'hex': '#e6e6fa'},
            {'name': 'Ruby Red', 'hex': '#9b111e'},
            {'name': 'Forest Green', 'hex': '#228b22'}
        ],
        'avoid': [
            {'name': 'Brown', 'hex': '#5c4033'},
            {'name': 'Khaki', 'hex': '#c3b091'},
            {'name': 'Olive', 'hex': '#808000'}
        ]
    },
    'medium': {  # MST5-6
        'recommended': [
            {'name': 'Coral', 'hex': '#ff7f50'},
            {'name': 'Turquoise', 'hex': '#40e0d0'},
            {'name': 'Olive Green', 'hex': '#556b2f'},
            {'name': 'Royal Blue', 'hex': '#4169e1'},
            {'name': 'Magenta', 'hex': '#c71585'}
        ],
        'avoid': [
            {'name': 'Neon Colors', 'hex': '#39ff14'},
            {'name': 'White', 'hex': '#ffffff'},
            {'name': 'Black', 'hex': '#000000'}
        ]
    },
    'medium deep': {  # MST7-8
        'recommended': [
            {'name': 'Gold', 'hex': '#ffd700'},
            {'name': 'Mustard Yellow', 'hex': '#ffdb58'},
            {'name': 'Orange', 'hex': '#ffa500'},
            {'name': 'Kelly Green', 'hex': '#4cbb17'},
            {'name': 'Electric Blue', 'hex': '#7df9ff'}
        ],
        'avoid': [
            {'name': 'Pastel Colors', 'hex': '#fadadd'},
            {'name': 'Beige', 'hex': '#f5f5dc'},
            {'name': 'Silver', 'hex': '#c0c0c0'}
        ]
    },
    'deep': {  # MST9-10
        'recommended': [
            {'name': 'Bright Yellow', 'hex': '#ffff00'},
            {'name': 'Fuchsia', 'hex': '#ff00ff'},
            {'name': 'Lime Green', 'hex': '#32cd32'},
            {'name': 'Bright Orange', 'hex': '#ff4500'},
            {'name': 'Aqua', 'hex': '#00ffff'}
        ],
        'avoid': [
            {'name': 'Dark Colors', 'hex': '#2f4f4f'},
            {'name': 'Brown', 'hex': '#5c4033'},
            {'name': 'Navy', 'hex': '#000080'}
        ]
    }
}

# Load model and class labels
def load_model_and_labels():
    try:
        # Load model
        model_path = r'models\mobilenetv2_mst_model94.h5'
        model = load_model(model_path)
        
        # Load class labels
        with open(r'models\mst_class_labels.json', 'r') as f:
            class_labels = json.load(f)
        
        return model, class_labels
    except Exception as e:
        print(f"Error loading model or labels: {e}")
        return None, None

# Face detection using OpenCV Haar Cascades
# Enhanced extract_skin_regions function with better detection for darker skin tones
def extract_skin_regions(face_img):
    if face_img is None:
        return None
    
    try:
        # Create a copy of the image
        img_ycrcb = face_img.copy()
        img_hsv = face_img.copy()
        
        # Convert to YCrCb color space - better for skin detection across diverse skin tones
        img_ycrcb = cv2.cvtColor(img_ycrcb, cv2.COLOR_BGR2YCrCb)
        
        # Define range for skin color in YCrCb (improved for darker skin tones)
        lower_skin_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
        upper_skin_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
        
        # Create a mask in YCrCb space
        mask_ycrcb = cv2.inRange(img_ycrcb, lower_skin_ycrcb, upper_skin_ycrcb)
        
        # Convert to HSV color space for additional detection
        img_hsv = cv2.cvtColor(img_hsv, cv2.COLOR_BGR2HSV)
        
        # Define range for skin color in HSV (widened for darker skin tones)
        lower_skin_hsv = np.array([0, 10, 30], dtype=np.uint8)
        upper_skin_hsv = np.array([25, 255, 255], dtype=np.uint8)
        
        # Create a mask in HSV space
        mask_hsv = cv2.inRange(img_hsv, lower_skin_hsv, upper_skin_hsv)
        
        # Combine the masks
        mask = cv2.bitwise_or(mask_ycrcb, mask_hsv)
        
        # Apply morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply the mask to extract skin regions
        skin = cv2.bitwise_and(face_img, face_img, mask=mask)
        
        return skin
    except Exception as e:
        print(f"Error in skin extraction: {e}")
        return None

# Function to detect face using multiple detectors
def detect_face_improved(img_path):
    try:
        # Load the cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Read the image
        img = cv2.imread(img_path)
        if img is None:
            return None, img, f"error_{int(time.time())}.jpg"
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Try to detect faces with different scaleFactor and minNeighbors values
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # If no faces detected, try with more lenient parameters
        if len(faces) == 0:
            faces = face_cascade.detectMultiScale(gray, 1.05, 3)
            
        # If still no faces detected, try with even more lenient parameters
        if len(faces) == 0:
            faces = face_cascade.detectMultiScale(gray, 1.03, 2)
        
        if len(faces) == 0:
            print("No faces detected, using entire image")
            # Create a unique filename with timestamp
            processed_filename = f"noface_{int(time.time())}.jpg"
            processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)
            # Save the original image as the processed image
            cv2.imwrite(processed_path, img)
            return None, img, processed_filename
        
        # Get coordinates of the largest face
        largest_face = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)[0]
        x, y, w, h = largest_face
        
        # Draw rectangle for visualization
        img_with_face = img.copy()
        cv2.rectangle(img_with_face, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Create a unique filename with timestamp
        processed_filename = f"processed_{int(time.time())}.jpg"
        processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)
        cv2.imwrite(processed_path, img_with_face)
        
        # Extract face region with a bit more margin
        y_margin = int(h * 0.1)
        x_margin = int(w * 0.1)
        y_start = max(0, y - y_margin)
        y_end = min(img.shape[0], y + h + y_margin)
        x_start = max(0, x - x_margin)
        x_end = min(img.shape[1], x + w + x_margin)
        
        face_region = img[y_start:y_end, x_start:x_end]
        
        return face_region, img_with_face, processed_filename
        
    except Exception as e:
        print(f"Error in face detection: {e}")
        processed_filename = f"error_{int(time.time())}.jpg"
        return None, cv2.imread(img_path), processed_filename


# Function to analyze skin tone with bias correction for darker skin tones
def analyze_skin_tone(model, class_labels, img_array, face_detected=True):
    try:
        # Get predictions from model
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = class_labels[predicted_class_idx]
        
        # Get the predicted MST number
        mst_num = int(predicted_class.replace('MST', ''))
        
        # Apply correction for darker skin tones
        # Get the average luminance of the image
        rgb_img = img_array[0] * 255  # Convert back from normalized
        # Simple conversion to grayscale to check luminance
        luminance = np.mean(0.299 * rgb_img[:,:,0] + 0.587 * rgb_img[:,:,1] + 0.114 * rgb_img[:,:,2])
        
        # If the image is dark (low luminance) but prediction is medium tone
        if luminance < 80 and 4 <= mst_num <= 7:
            # Adjust prediction to darker range
            adjusted_mst_num = min(10, mst_num + 2)
            predicted_class = f"MST{adjusted_mst_num}"
            
        # If very dark skin (very low luminance) but not predicted as dark
        if luminance < 50 and mst_num < 8:
            # Adjust prediction to darkest range
            adjusted_mst_num = min(10, mst_num + 3)
            predicted_class = f"MST{adjusted_mst_num}"
        
        return predicted_class, confidence
    except Exception as e:
        print(f"Error in skin tone analysis: {e}")
        return "MST5", 0.5  # Default moderate value with low confidence

# Preprocess image for the model
def preprocess_image(img_array, target_size=(224, 224)):
    try:
        # Convert BGR to RGB if the input is from OpenCV
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        # Resize the image
        img = Image.fromarray(img_array)
        img = img.resize(target_size)
        
        # Convert to array and normalize
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        return img_array
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None

# Get skin tone group based on MST index
def get_skin_tone_group(mst_index):
    index = int(mst_index.replace('MST', ''))
    if 1 <= index <= 2:
        return 'light'
    elif 3 <= index <= 4:
        return 'light medium'
    elif 5 <= index <= 6:
        return 'medium'
    elif 7 <= index <= 8:
        return 'medium deep'
    else:  # 9-10
        return 'deep'
    
@app.route('/')
def index():
    return render_template('index.html', mst_colors=MST_COLORS)

@app.route('/static/images/<path:filename>')
def serve_images(filename):
    return send_from_directory(IMAGES_FOLDER, filename)

@app.route('/static/processed/<path:filename>')
def serve_processed(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        try:
            # Generate a unique filename with timestamp to avoid overwriting
            timestamp = int(time.time())
            file_extension = os.path.splitext(file.filename)[1]
            unique_filename = f"upload_{timestamp}{file_extension}"
            
            # Save the uploaded file with the unique name
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            # Optional: Clean up old processed images (keeping the 20 most recent)
            try:
                processed_files = sorted([os.path.join(PROCESSED_FOLDER, f) for f in os.listdir(PROCESSED_FOLDER)], 
                                         key=os.path.getmtime)
                if len(processed_files) > 20:  # Keep only the 20 most recent files
                    for old_file in processed_files[:-20]:
                        os.remove(old_file)
            except Exception as e:
                print(f"Error cleaning processed files: {e}")
            
            # Load model and labels if not already loaded
            model, class_labels = load_model_and_labels()
            if model is None or class_labels is None:
                return jsonify({'error': 'Failed to load model or class labels'})
            
            # Detect face using improved method
            face_region, img_with_face, processed_filename = detect_face_improved(file_path)
            
            # If face was detected, use it for prediction
            if face_region is not None:
                # Extract skin regions from the face with improved method
                skin_region = extract_skin_regions(face_region)
                
                # If skin extraction worked, use it; otherwise use the whole face
                if skin_region is not None and np.sum(skin_region) > 0:
                    # Check if we have enough skin pixels to analyze
                    non_zero_pixels = cv2.countNonZero(cv2.cvtColor(skin_region, cv2.COLOR_BGR2GRAY))
                    if non_zero_pixels > 100:  # Arbitrary threshold
                        img_for_prediction = skin_region
                    else:
                        img_for_prediction = face_region
                else:
                    img_for_prediction = face_region
            else:
                # If no face detected, use the entire image
                img = cv2.imread(file_path)
                img_for_prediction = img
            
            # Preprocess image
            img_array = preprocess_image(img_for_prediction)
            
            if img_array is None:
                return jsonify({'error': 'Failed to preprocess image'})
            
            # Analyze skin tone with improved method
            predicted_class, confidence = analyze_skin_tone(model, class_labels, img_array, face_region is not None)
            
            # Get color hex code for the predicted MST
            mst_index = int(predicted_class.replace('MST', '')) - 1
            mst_color = MST_COLORS[mst_index]
            
            # Get skin tone group and clothing recommendations
            skin_tone_group = get_skin_tone_group(predicted_class)
            recommendations = CLOTHING_RECOMMENDATIONS[skin_tone_group]
            
            # Add timestamp to URLs to prevent caching
            image_url = url_for('static', filename=f'uploads/{unique_filename}')
            processed_image_url = url_for('static', filename=f'processed/{processed_filename}')
            
            # Create response
            response = {
                'success': True,
                'prediction': predicted_class,
                'confidence': confidence * 100,  # Convert to percentage
                'mst_color': mst_color,
                'mst_index': mst_index,
                'image_url': f"{image_url}?t={timestamp}",
                'processed_image_url': f"{processed_image_url}?t={timestamp}",
                'face_detected': face_region is not None,
                'recommendations': recommendations,
                'skin_tone_group': skin_tone_group,
                'timestamp': timestamp  # Include timestamp in response
            }
            return jsonify(response)
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Error during processing: {str(e)}'})
    
    return jsonify({'error': 'Failed to process image'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Server starting on http://localhost:{port}")
    serve(app, host='0.0.0.0', port=port)
