# ✊✋✌️ Rock-Paper-Scissors Image Classification Using CNN

## 📌 Project Overview  
This project classifies hand gestures into **Rock**, **Paper**, or **Scissors** categories using **Convolutional Neural Networks (CNNs)**.  

To better understand performance differences, we first implemented traditional **Machine Learning (ML)** and **Artificial Neural Network (ANN)** approaches. These methods were limited in generalization capability for unseen image data, as they rely heavily on manually extracted features.  

This served as a **baseline comparison** to highlight the superior performance of CNNs, which can automatically learn spatial hierarchies from raw images. CNNs outperformed ML and ANN models significantly in terms of accuracy and robustness.

---

<img src="Images/Stone-Paper-Scissors.png" alt="Stone Paper Scissors Game Demo" width="600"/>

---

## 🗂️ Dataset Overview  
We used an image dataset containing labeled hand gesture images:
- `rock/` – images of a closed fist  
- `paper/` – images of an open hand  
- `scissors/` – images of two extended fingers  

## 🗂️ Dataset Overview  
We used an image dataset containing labeled hand gesture images:
- `rock/` – images of a closed fist  
- `paper/` – images of an open hand  
- `scissors/` – images of two extended fingers  

> ⚠️ The dataset is not included in this repository due to size limits.  
> Please use your own or download a public Rock-Paper-Scissors image dataset and structure it as shown above.  
> 📩 *If you would like access to the dataset used in this project, feel free to contact me at [uqashazahid@gmail.com](mailto:uqashazahid@gmail.com).*

---

Dataset/
├── train/
│ ├── rock/
│ ├── paper/
│ └── scissors/
├── validation/
│ ├── rock/
│ ├── paper/
│ └── scissors/
└── test/
├── rock/
├── paper/
└── scissors/

---

## 📁 Files Overview  

The `CNN/` folder contains all files related to the Convolutional Neural Network implementation:

- **`cnn.ipynb`** – This Jupyter Notebook includes all code for:
  - Training the CNN model  
  - Performing real-time predictions  
  - Evaluating the model  
  - Visualizing performance metrics  

- **Saved Keras model files** – After training, the best-performing models (based on accuracy) are saved in this folder for later use.

- **Image Augmentation** – We applied various image augmentation techniques (e.g., rotation, zoom, flip) to artificially expand the training dataset. This helped demonstrate how:
  - Augmentation increases dataset diversity  
  - Accuracy improves  
  - Overfitting is reduced  

These files showcase both the process and results of training robust image classification models using CNNs.


