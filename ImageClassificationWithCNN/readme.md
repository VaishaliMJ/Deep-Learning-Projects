## 📖  Project Description
This project classify handwritten digits(0-9) from MNIST dataset. System is build and trained using Convolutional Neural Network(CNN) The model is trained on the standard MNIST dataset, which contains 60,000 training images and 10,000 testing images, each a 28x28 pixel grayscale image.

---
### 📊 Dataset Information
MNIST is a standard sklearn dataset.It is a foundational collection of 70,000 grayscale images (60k training, 10k testing) of handwritten digits (0-9), each 28x28 pixels
*  Handwritten labelled digits 0 through 9.
*  70,000 images including training and test data
*  Images are 28*28 of grey-scale
---  
### ⚙️ Model Architecture
Model consist of following layers
*  **Convolutional Layers**:Applied filters of size 32 and 64 to extract features like edges
*  **Activation Function**:Relu is activation function
*  **BatchNormalization**:Batch normalization is used after each layer to normalize input
*  **Pooling Layers**: Max pooling layers are used to downsample the feature maps
*  **Dropout**:Added to avoid overfitting
*  **Flatten Layer**: Converts the 2D feature maps into a 1D
*  **Dense Layer**:To prevent overfitting used softmax activation function
*  **Optimizer**:Adam optimizer is used during training phase
---
### 🛠️ Prerequisites
Install libraries  tensorflow keras numpy matplotlib

---
### 📂 Expected Outputs
Evaluated model using accuracy score, confusion matrix and loss curves
Visualised predictions with Matplotlib to compare correctly vs misclassified digits

---
#### ✍️ Author
 Vaishali M. Jorwekar<br>
 Date	:20 Jan 2026





