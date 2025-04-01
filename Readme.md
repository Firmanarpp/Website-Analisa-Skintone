# SkinToneAI - Skin Tone Analyzer

SkinToneAI is a sophisticated web application that analyzes your skin tone using a deep learning model and provides personalized clothing color recommendations based on the Monk Skin Tone (MST) scale.

![SkinToneAI Screenshot](screenshot.png)

## Features

- **AI-Powered Skin Tone Analysis**: Upload a photo to detect your skin tone classification on the MST scale
- **Personalized Recommendations**: Get clothing color suggestions that complement your skin tone
- **Modern, Responsive UI**: Beautiful user interface that works across all devices
- **Detailed Visualization**: Interactive MST scale to understand your skin tone classification

## Technology Stack

- **Frontend**: HTML5, CSS3, JavaScript
- **Backend**: Flask (Python)
- **Machine Learning**: TensorFlow, MobileNetV2
- **Data Visualization**: Custom SVG graphics and CSS animations

## Project Structure

```
skin-tone-analyzer/
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── script.js
│   ├── images/
│   │   └── hero-image.svg
│   └── uploads/      (folder for uploaded images)
├── templates/
│   └── index.html
├── models/
│   ├── mobilenetv2_mst_model.h5
│   └── mst_class_labels.json
├── app.py
├── hero-image.svg
└── requirements.txt
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/skin-tone-analyzer.git
   cd skin-tone-analyzer
   ```

2. **Create and activate a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install the dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the model files**
   - Place your `mobilenetv2_mst_model.h5` and `mst_class_labels.json` files in the `models/` directory

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the website**
   - Open your browser and navigate to `http://127.0.0.1:5000/`

## Usage

1. **Browse the website** to learn about skin tones and the MST scale
2. **Upload an image** by clicking on the upload area or dragging and dropping a file
3. **Click "Analyze Skin Tone"** to process your image
4. **View results** showing your skin tone classification on the MST scale
5. **Explore recommendations** for clothing colors that complement your skin tone

## Model Information

The skin tone classifier uses a MobileNetV2 architecture that has been fine-tuned on a dataset of diverse skin tones. The model classifies skin tones according to the Monk Skin Tone (MST) Scale, which provides a more inclusive representation of human skin diversity than traditional scales.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- The Monk Skin Tone Scale developed by Dr. Ellis Monk
- TensorFlow and Keras for making deep learning accessible
- Flask for the simple and powerful web framework