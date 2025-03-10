## 🌱 Plant Disease Prediction Using CNN  
A **deep learning-based web application** built with **Streamlit** that detects plant diseases from leaf images using a **Convolutional Neural Network (CNN)** model.

---

## 📌 Features  
✅ Upload an image of a plant leaf for disease classification  
✅ Uses a pre-trained **CNN model (.h5 file)** for predictions  
✅ Provides real-time results with **high accuracy**  
✅ Web-based interface built with **Streamlit**  

---

## ⚙️ Technologies Used  
- **Python** (Machine Learning & Deep Learning)  
- **TensorFlow/Keras** (CNN Model)  
- **Streamlit** (Frontend for deployment)  
- **NumPy & PIL** (Image processing)  
- **gdown** (For downloading model from Google Drive)  

---

## 📂 Folder Structure  

```
📁 Plant-Disease-Prediction
│-- 📁 Model
│   │-- plant_disease_prediction_model.h5  # CNN Model
│   │-- class_indices.json  # Class Labels
│-- app.py  # Main Streamlit App
│-- requirements.txt  # Dependencies
│-- README.md  # Project Documentation
|-- start.sh   #For deployment
```

---

## 🚀 Installation & Setup  

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/Mohan0416/Plant-Disease-Prediction-Using-CNN.git
cd Plant-Disease-Prediction-Using-CNN
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Run the Application**
```bash
streamlit run app.py
```

---

## 🖼️ Sample Prediction  
**Input:**  
![input](https://github.com/user-attachments/assets/4091ff24-7f31-4a10-b597-ab29118903e4)
 

**Output:**  
![output](https://github.com/user-attachments/assets/03ad5636-1d01-42ff-a6d1-35049a3e5e4f)

---
