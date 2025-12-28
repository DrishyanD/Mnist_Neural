# Handwritten Digit Recognition Using a 2-Layer Neural Network

**Author:** Drishyan D  

This project implements a **simple neural network from scratch** in Python to recognize handwritten digits (0–9) using **NumPy**.  
It demonstrates the core concepts of neural networks, including **forward propagation**, **backward propagation**, **gradient descent**, and **activation functions** (ReLU and Softmax), without using high-level frameworks like TensorFlow or PyTorch.

---

## **Data**

- **Dataset:** MNIST handwritten digits  
- **Source:** [https://www.kaggle.com/datasets/oddrationale/mnist-in-csv](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)  
- **Files used:** `mnist_train.csv`, `mnist_test.csv` (not included in the repo due to GitHub size limits)  
- **Format:** Each CSV contains 28×28 grayscale images, with the first column as the label (0–9) and the rest as pixel values (0–255).

> **Note:** You need to download the CSV files and place them in the `data/` folder locally. This folder is ignored in GitHub (`.gitignore`) to avoid pushing large files.

---

## **Project Structure**
```text
Mnist_Neural/
│
├─ data/ # Folder for CSV dataset files (ignored in Git)
├─ main.ipynb # Jupyter Notebook with neural network code
├─ .gitignore # Ignores data, venv, checkpoints, etc.
├─ venv/ # Python virtual environment (ignored)
```

---

## **Neural Network Overview**

- **Input layer:** 784 neurons (28×28 pixels per image)  
- **Hidden layer:** 10 neurons, **ReLU activation**  
- **Output layer:** 10 neurons (digits 0–9), **Softmax activation**  

**How it works (step-by-step):**

1. **Prepare data:** Flatten images and normalize pixel values to 0–1.  
2. **Initialize weights and biases:** Random values for W1, W2, b1, b2.  
3. **Forward propagation:** Compute hidden and output layer activations.  
4. **Compute loss:** Cross-entropy loss measures prediction error.  
5. **Backward propagation:** Calculate gradients for weights and biases.  
6. **Update parameters:** Apply gradient descent to improve predictions.  
7. **Make predictions:** Use trained network to classify digits.  
8. **Evaluate accuracy:** Compare predicted labels to true labels.  

---

## **Installation & Usage**

1. **Clone the repo:**
```bash
git clone https://github.com/DrishyanD/Mnist_Neural.git
```
Install dependencies:
```bash
pip install numpy matplotlib
```
- Download MNIST CSV files from Kaggle and place them in data/.
- Run the Notebook:
Open main.ipynb in Jupyter Notebook and run all cells to train and test the network.
- Output
Training & Development Accuracy
- Sample Predictions: Visualizations showing images with predicted and true labels.

## **Requirements**
- Python 3.x  
- NumPy  
- Matplotlib


## **Credits / References**

**Acknowledgements**
- MNIST dataset provided by Kaggle.

**Tutorial Author:** Samson Zhang  
- Shared a **Simple MNIST Neural Network from scratch** using **NumPy** (no TensorFlow/Keras).  
- Video link: [YouTube Tutorial](https://www.youtube.com/watch?v=w8yWXqWQYmU)

**Notes:**  
- His tutorial inspired the implementation and structure of this project.  
- All code in this repo is written independently, but the learning approach is credited to him.
