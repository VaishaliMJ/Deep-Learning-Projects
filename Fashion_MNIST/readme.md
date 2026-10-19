## 📖  Project Description
This project classifies images FASHION MNIST dataset. System is build and trained using Convolutional Neural Network(CNN) The model is trained on the standard FASHION MNIST dataset, which contains 70,000 training images and 10,000 testing images, each a 28x28 pixel grayscale image.

### 📊 Dataset Information
FASHION MNIST is a standard sklearn dataset.It is a foundational collection of 70,000 grayscale images (60k training, 10k testing) of clothing images, each 28x28 pixels
*  Cloth images of 10 items.
*  70,000 images including training and test data
*  Images are 28*28 of grey-scale
*  Class Labels
        1. T-shirt/top
        2. Trouser
        3. Pullover
        4. Dress
        5. Coat
        6. Sandal
        7. Shirt
        8. Sneaker
        9. Bag
        10. Ankle boots
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
Visualised predictions with Matplotlib to compare correctly vs misclassified cloth images

---
#### ✍️ Author
 Vaishali M. Jorwekar<br>
 Date	:20 Jul 2026          
