"""---------------------------------------------------------------------------------------------------------------
                        Surface Crack Detction Using CNN 
                            Test Module       
                            Vaishali Jorwekar
-------------------------------------------------------------------------------------------------------------------
Problem statement:Develope an AI-Based Surface Crack Detection System using 
                 Convolutional Neural Networks (CNN) to identify cracks on 
                 industrial surfaces from image datasets. 
-------------------------------------------------------------------------------------------------------------------"""
#####################################################################################################
#   Imports and constants
#####################################################################################################
from SurfaceCrackDetectionUsingCNN import PROCESSED_DATASET,IMAGE_SIZE,getValidImageFiles
import os 
import numpy as np
import matplotlib.pyplot as plt
import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import random
FINAL_MODEL_NAME="Final_Marvellous_Crack_Detection_Model.keras"
BEST_MODEL="Best_Crack_Detection_Model.keras"


#########################################################################################################
#   Function Name    :  loadImage
#   Description      :  load image from a path
#   Input Params     :  -   
#   Output Params    :  -
#   Author           :  Vaishali M Jorwekar
#   Date             :  29 May 2026
#########################################################################################################
def predictSingleImage(imagePath):
    if not os.path.exists(imagePath):
        print("ERROR: Image not found:", imagePath)
        return

    img = load_img(imagePath, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    model=keras.models.load_model(BEST_MODEL)

    prediction = model.predict(img_array)
    prediction_value = prediction[0][0]

    #crack_index = train_data.class_indices["Crack"]
    crack_index=["Crack","No Crack"]
    if crack_index == 1:
        final_result = "Crack Detected" if prediction_value > 0.5 else "No Crack"
    else:
        final_result = "No Crack" if prediction_value > 0.5 else "Crack Detected"

    print("=" * 70)
    print("Single Image Prediction")
    print("=" * 70)
    print("Image Path       :", imagePath)
    print("Prediction Value :", prediction_value)
    print("Final Result     :", final_result)

    plt.imshow(load_img(imagePath))
    plt.title(final_result)
    plt.axis("off")
    plt.show()
#########################################################################################################
#   Function Name    :  main function 
#   Description      :  main function,manages calls to other functions
#   Input Params     :  -   
#   Output Params    :  -
#   Author           :  Vaishali M Jorwekar
#   Date             :  29 May 2026
#########################################################################################################
def main():
    sample_test_image = os.path.join(PROCESSED_DATASET, "test", "Crack")

    test_images = getValidImageFiles(sample_test_image)

    if len(test_images) > 0:
        predictSingleImage(os.path.join(sample_test_image, test_images[5]))
    else:
        print("No test image found for single image prediction.")

#############################################################################################
#   Starter
##############################################################################################
if __name__=="__main__":
    main()


