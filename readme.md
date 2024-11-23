# 🐕 Dog Breed Classification with Deep Learning

A Streamlit web application that identifies dog breeds from images using a deep learning model based on MobileNetV2. The model is trained to recognize multiple dog breeds with high accuracy.

## 🚀 Features

- Real-time dog breed classification from uploaded images
- Support for multiple image formats (JPG, PNG)
- Interactive breed search functionality
- Confidence scores for predictions
- Visual representation of top 10 predicted breeds
- User-friendly interface with Streamlit

## 📋 Prerequisites

```bash
python >= 3.8
streamlit
tensorflow
tensorflow-hub
pillow
numpy
plotly
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dog-breed-classifier.git
cd dog-breed-classifier
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the model:
- The trained model should be placed in the `Model` directory
- Ensure the model file is named: `-1000-images-mobilenetv2-Adam`

4. Prepare breed labels:
- Place `breed_labels.json` in the `Data` directory

## 💻 Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the displayed local URL (typically `http://localhost:8501`)

3. Use the application:
   - Search for supported dog breeds using the search box
   - Upload a dog image using the file uploader
   - View predictions and confidence scores
   - Explore the top 10 predicted breeds in the interactive chart

## 🤖 Model Training

The model was trained using transfer learning with MobileNetV2 as the base model. You can find the complete training process in the [Jupyter notebook](Notebooks/Dog_Breed_Classification.ipynb).

### Training Details
- Base Model: MobileNetV2
- Optimizer: Adam
- Input Size: 224x224
- Training Dataset: [Specify your dataset source]
- Number of Classes: [Specify number of breeds]
- Training Accuracy: [Specify accuracy]
- Validation Accuracy: [Specify accuracy]

## 📁 Project Structure

```
dog-breed-classifier/
├── app.py                  # Streamlit application
├── requirements.txt        # Project dependencies
├── Data/
│   └── breed_labels.json   # Mapping of class indices to breed names
├── Model/
│   └── -1000-images-mobilenetv2-Adam/  # Trained model files
└── Notebooks/
    └── Dog_Breed_Classification.ipynb   # Training notebook
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Tensorflow](https://www.tensorflow.org/) for the deep learning framework
- [Streamlit](https://streamlit.io/) for the web application framework
- [MobileNetV2](https://arxiv.org/abs/1801.04381) paper and implementation
- [Dataset source and any other resources you used]

## 📧 Contact

Your Name - [Your Email]

Project Link: [https://github.com/yourusername/dog-breed-classifier](https://github.com/yourusername/dog-breed-classifier)
