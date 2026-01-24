## Real Time Emotion Detection(CNN+OpenCV) 😊😑😔😡😯😨🥸
Developed a real-time facial emotion recognition system using convolutional Neural Networks (CNN) integrated with OpenCV for live video capture.The model successfully detects and classifies human emotions such as happy, sad, angry, supervised and neutral from webcam input 

---

### ⚙️ Model Architecture
Model consist of following layers
*  **Convolutional Layers**:Applied different filters of sizes like  32,64... to extract features like edges
*  **Activation Function**:Relu is activation function
*  **BatchNormalization**:Batch normalization is used after each layer to normalize input
*  **Pooling Layers**: Max pooling layers are used to downsample the feature maps
*  **Dropout**:Added to avoid overfitting
*  **Flatten Layer**: Converts the 2D feature maps into a 1D
*  **Dense Layer**:To prevent overfitting used softmax activation function
*  **Optimizer**:Adam optimizer is used during training phase
---

### 📊 Dataset Information
*  FER - 2013 dataset with 7 emotion types
*  Dataset contains two seperate folders for train and test data set
*  Both folder contains seperate folders for 7 emotion images as angry,disgusted,fearful,happy,neutral,sad,suprised
*  48x48 pixel grayscale image

---
### 🛠️ Prerequisites
Install libraries  tensorflow keras numpy matplotlib


---
### 📂 Expected Outputs
*  **Evaluated model using accuracy score, confusion matrix and loss curves
*  **Visualised predictions with Matplotlib to compare correctly vs misclassified emotions
*  **Tested model with live video 

---
#### ✍️ Author
 Vaishali M. Jorwekar<br>
 Date	:24 Jan 2026


