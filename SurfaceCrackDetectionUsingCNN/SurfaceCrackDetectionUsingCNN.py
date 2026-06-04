"""---------------------------------------------------------------------------------------------------------------
                Surface Crack Detction Using CNN
                            Vaishali Jorwekar
-------------------------------------------------------------------------------------------------------------------
Problem statement:Develope an AI-Based Surface Crack Detection System using 
                 Convolutional Neural Networks (CNN) to identify cracks on 
                 industrial surfaces from image datasets. 
-------------------------------------------------------------------------------------------------------------------"""
#####################################################################################################
#   Imports and constants
#####################################################################################################
import numpy as np
import argparse
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import os
import shutil
import keras
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix,classification_report,ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import load_img, img_to_array

#####################################################################################################
#   Constants and file names
#####################################################################################################
BORDER="-"*65
RANDOM_SEED = 42
ORIGINAL_DATASET = "CrackDataset"
POSITIVE_FOLDER = os.path.join(ORIGINAL_DATASET, "Positive")
NEGATIVE_FOLDER = os.path.join(ORIGINAL_DATASET, "Negative")


PROCESSED_DATASET = "Processed_CrackDataset"

OUTPUT_DIR="Output_Saved"

IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15

FINAL_MODEL_NAME=os.path.join(OUTPUT_DIR,"Final_Marvellous_Crack_Detection_Model.keras")
BEST_MODEL=os.path.join(OUTPUT_DIR,"Best_Crack_Detection_Model.keras")

###########################################################################################
#   Function        :   checkFolder
#   Input Params    :   path(str)-directory path
#   Output Params   :   None
#   Description     :   Checks if input images folder found
#   Author          :   Vaishali M Jorwekar
#   Date            :   29 May 2026
############################################################################################
def checkFolder(path:str):
    if not os.path.exists(path):
        print("ERROR: Folder not found:", path)
        exit()
###########################################################################################
#   Function        :   getValidImageFiles
#   Input Params    :   path
#   Output Params   :   Images with valid extensions
#   Description     :   Check if image folder contains valid images
#   Author          :   Vaishali M Jorwekar
#   Date            :   29 May 2026
############################################################################################        
def getValidImageFiles(folder):
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return [
        file for file in os.listdir(folder)
        if file.lower().endswith(valid_extensions)
    ]  
#########################################################################################################
#   Function Name    :  createIndustrialFolderStructure
#   Description      :  Create Industrial Folder Structure
#   Input Params     :  -   
#   Output Params    :  Folders
#   Author           :  Vaishali M Jorwekar
#   Date             :  29 May 2026
#########################################################################################################
def createIndustrialFolderStructure():
    folders = [
                "train/Crack",
                "train/NoCrack",
                "validation/Crack",
                "validation/NoCrack",
                "test/Crack",
                "test/NoCrack"
             ]   
    for folder in folders:
        os.makedirs(os.path.join(PROCESSED_DATASET, folder), exist_ok=True)
#########################################################################################################
#   Function Name    :  splitDataSetCopyImages
#   Description      :  Splits Data set into training,validation and test 
#   Input Params     :  sourceFolder,imageFiles,folderName 
#   Output Params    :  splitted data set into training,validation and test folders
#   Author           :  Vaishali M Jorwekar
#   Date             :  29 May 2026
######################################################################################################### 
def splitDataSetCopyImages(sourceFolder, imageFiles,imageType):
    random.shuffle(imageFiles)
    totalImages = len(imageFiles)
    
    #   Splitting training images into training and validation sets
    trainImagesCount=int(totalImages*0.70)
    validationImageCount=int(totalImages*0.15)
    
    trainFiles=imageFiles[:trainImagesCount]

    validationFiles=imageFiles[trainImagesCount:trainImagesCount+validationImageCount]
    testFiles=imageFiles[trainImagesCount + validationImageCount:]

    splitDataSet={
        "train":trainFiles,
        "validation":validationFiles,
        "test":testFiles
    }
     
    for folderName,files in splitDataSet.items():
        destinationFolder=getDestinationFolder(imageType, folderName)
       
        for file in files:
            source_path = os.path.join(sourceFolder, file)
            destination_path = os.path.join(destinationFolder, file)

            if not os.path.exists(destination_path):
                shutil.copy(source_path, destination_path)


    
    print(f"{imageType} Images Split:")
    print(f"Training Images   : {len(trainFiles)}")
    print(f"Validation Images :{len(validationFiles)}")
    print(f"Testing Images    :{len(testFiles)}")
    print(BORDER)
#########################################################################################################
#   Function Name    :  getDestinationFolder
#   Description      :  Creates destination folder as Crack or NoCrack
#   Input Params     :  imageType, folderName
#   Output Params    :  Newly created folders
#   Author           :  Vaishali M Jorwekar
#   Date             :  29 May 2026
######################################################################################################### 
def getDestinationFolder(imageType, folderName):
    #print(f"imageType:{imageType}")
    #print(f"folderName:{folderName}")
    return os.path.join(
            PROCESSED_DATASET,
            folderName,
            imageType
        )

#########################################################################################################
#   Function Name    :  gettTrainTestValidationDir
#   Description      :  Return Train,Test and Validationdir path
#   Input Params     :  -   
#   Output Params    :  train,test and validation directory path
#   Author           :  Vaishali M Jorwekar
#   Date             :  29 May 2026
#########################################################################################################
def gettTrainTestValidationDir():
    trainDir = os.path.join(PROCESSED_DATASET, "train")
    validationDir = os.path.join(PROCESSED_DATASET, "validation")
    testDir = os.path.join(PROCESSED_DATASET, "test")  
    return trainDir,validationDir,testDir  
#########################################################################################################
#   Function Name    :  imagePreprocessing
#   Description      :  Dislay Sample Images
#   Input Params     :  trainData 
#   Output Params    :  -
#   Author           :  Vaishali M Jorwekar
#   Date             :  30 May 2026
#########################################################################################################
def displaySampleImages(trainData):
    sample_images, sample_labels = next(trainData)

    plt.figure(figsize=(10, 6))

    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.imshow(sample_images[i])

        

        if sample_labels[i] == trainData.class_indices["Crack"]:
            plt.title("Crack")
        else:
            plt.title("No Crack")

        plt.axis("off")

    plt.suptitle("Marvellous CNN Sample Training Images")
    #plt.show()
    plt.savefig(os.path.join(OUTPUT_DIR,"SampleImages.png"))
    plt.close() 
        
###########################################################################################
#   Function        :   save_summary
#   Input Params    :   test_acc(float),test_loss(float),epochs(int),out_dir(str)
#   Output Params   :   Saves summary text file
#   Description     :   Saves final model performance summary and artifact list
#   Author          :   Vaishali M Jorwekar
#   Date            :   2 Jun 2026
##########################################################################################
def save_summary(test_acc,test_loss,epochs,out_dir=OUTPUT_DIR):
    ensure_dir(out_dir)
    with open(os.path.join(out_dir,"summary.txt"),"w") as f:
        f.write(
            "Surface CrackDetection CNN Summary\n"
            "================================================================\n"
            f"Test Accuracy :{test_acc:.4f}\n"
            f"Test Loss     :{test_loss:.4f}\n"
            f"Epochs    :{epochs}\n"
            f"OUTPUT Saved:TrainingAccuracy.png,TrainingLoss.png,SampleImages.png\n"
            "confusion_matrix.png,classification_report\n"
            "Best_Crack_Detection_Model.keras,Final_Marvellous_Crack_Detection_Model.keras"

        )    
#########################################################################################################
#   Function Name    :  imagePreprocessing
#   Description      :  Image Pre-processing and Augumentation
#   Input Params     :  trainDir,validationDir,testDir   
#   Output Params    :  -
#   Author           :  Vaishali M Jorwekar
#   Date             :  29 May 2026
#########################################################################################################
def imagePreprocessing(trainDir,validationDir,testDir):
    
    
    trainImgDataGen=ImageDataGenerator(rescale=1.0 / 255,
                                       rotation_range=15,
                                       zoom_range=0.2,
                                       width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        horizontal_flip=True
                                        )
    
    
    validationImageDataGen = ImageDataGenerator(rescale=1.0 / 255)

    testImageDataGen = ImageDataGenerator(rescale=1.0 / 255)
    
    trainData = trainImgDataGen.flow_from_directory(
                                trainDir,
                                target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                batch_size=BATCH_SIZE,
                                class_mode="binary"
                                )
    
    
    validationData = validationImageDataGen.flow_from_directory(
                        validationDir,
                        target_size=(IMAGE_SIZE, IMAGE_SIZE),
                        batch_size=BATCH_SIZE,
                        class_mode="binary"
                    )

    testData = testImageDataGen.flow_from_directory(
                            testDir,
                            target_size=(IMAGE_SIZE, IMAGE_SIZE),
                            batch_size=BATCH_SIZE,
                            class_mode="binary",
                            shuffle=False
                        )
    print("Class Indices:", trainData.class_indices)
    #   Display Sample images
    displaySampleImages(trainData)
    return trainData,validationData,testData
#########################################################################################################
#   Function Name    :  buildCNNModel
#   Description      :  Builds CNN Model
#   Input Params     :  -   
#   Output Params    :  Built CNN Model
#   Author           :  Vaishali M Jorwekar
#   Date             :  29 May 2026
#########################################################################################################
def buildCNNModel():  
    model=Sequential()

    model.add(Conv2D(filters=32,kernel_size=(3,3),activation="relu",input_shape=(IMAGE_SIZE,IMAGE_SIZE,3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(filters=64,kernel_size=(3,3),activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=128,kernel_size=(3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    
    model.add(Dense(units=256, activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(units=128, activation="relu"))
    model.add(Dropout(0.3))

    model.add(Dense(units=1, activation="sigmoid"))
    
    model.compile(
                optimizer="adam",
                loss="binary_crossentropy",
                metrics=["accuracy"]
            )

    model.summary()
    return model
#########################################################################################################
#   Function Name    :  trainCNNModel
#   Description      :  Train CNN Model
#   Input Params     :  -   
#   Output Params    :  Trained CNN Model
#   Author           :  Vaishali M Jorwekar
#   Date             :  29 May 2026
#########################################################################################################
def trainCNNModel(model,trainData,validationData): 
    earlyStop = EarlyStopping(
                monitor="val_loss",
                patience=4,
                restore_best_weights=True
                )
    checkpoint = ModelCheckpoint(
                    BEST_MODEL,
                    monitor="val_accuracy",
                    save_best_only=True,
                    mode="max",
                    verbose=1
                )
    reduceLR = ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.2,
                patience=2,
                min_lr=0.00001,
                verbose=1
                )   
    
    history = model.fit(
                    trainData,
                    epochs=EPOCHS,
                    validation_data=validationData,
                    callbacks=[earlyStop, checkpoint, reduceLR]
                    )
    
    return history
#########################################################################################################
#   Function Name    :  evaluateModel
#   Description      :  Test Model on new data
#   Input Params     :  model,testData   
#   Output Params    :  -
#   Author           :  Vaishali M Jorwekar
#   Date             :  30 May 2026
#########################################################################################################
def evaluateModel(model,testData):
    print(BORDER)
    print("Testing Model on Unseen Test Data")
    print(BORDER)

    test_loss, test_accuracy = model.evaluate(testData)

    print(f"Test Loss     : {test_loss}")
    print(f"Test Accuracy : {test_accuracy * 100}")
    
    return test_loss, test_accuracy

#########################################################################################################
#   Function Name    :  testModel
#   Description      :  Test and predict Model on new data
#   Input Params     :  model,testData   
#   Output Params    :  -
#   Author           :  Vaishali M Jorwekar
#   Date             :  30 May 2026
#########################################################################################################
def testModel(model,testData):
    predictions = model.predict(testData)
    predicted_classes = (predictions > 0.5).astype(int).reshape(-1)

    actual_classes = testData.classes
    cm=confusion_matrix(actual_classes, predicted_classes)
    print("Confusion Matrix:")
    print(cm)
    class_names = list(testData.class_indices.keys())

    displayCM=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=class_names)
    displayCM.plot(cmap=plt.cm.Blues)
    
    plt.savefig(os.path.join(OUTPUT_DIR,'confusion_matrix.png'), dpi=300, bbox_inches='tight')

    plt.close()
    target_names=list(testData.class_indices.keys())

    show_misclassifications(testData,actual_classes,predicted_classes,target_names)

    report=classification_report(
        actual_classes,
        predicted_classes,
        target_names=list(testData.class_indices.keys())
    )
    print(f"Classification Report:\n{report}")
    
    with open (os.path.join(OUTPUT_DIR,"classification_report.txt"),"w") as f:
        f.write(report)
        
    model.save(FINAL_MODEL_NAME)

    print(f"Final model saved successfully.{FINAL_MODEL_NAME}")
###########################################################################################
#   Function        :   show_misclassifications
#   Input Params    :   actual_classes,predicted_classes,target_names
#   Output Params   :   Saves misclassification grid
#   Description     :   Displays and saves examples of misclassified test images
#   Author          :   Vaishali M Jorwekar
#   Date            :   2 Jun 2026
##########################################################################################
def show_misclassifications(testData,actual_classes,predicted_classes,target_names,limit=15):
    wrong=np.where(actual_classes!=predicted_classes)[0]
    print(len(wrong))
    if(len(wrong)==0):
        print("No Misclassifications !!!....")
        return 
    sel=wrong[:limit]
    cols=3
    rows=int(np.ceil(len(sel)/cols))
    plt.figure(figsize=(12,2.6*rows))
    for i,idx in enumerate(sel,1):
        imagePath = testData.filepaths[idx]
        filename = os.path.basename(imagePath) 
        plt.subplot(rows, cols, i)
        plt.imshow(load_img(imagePath))
        plt.title(f"T:{target_names[int(actual_classes[idx])]} \
                  \nP:{target_names[int(predicted_classes[idx])]}\
                      \n{filename}", fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"missclassification.png"))
    plt.close() 
#########################################################################################################
#   Function Name    :  plotGraphs
#   Description      :  Plot Graphs
#   Input Params     :  history   
#   Output Params    :  -
#   Author           :  Vaishali M Jorwekar
#   Date             :  29 May 2026
#########################################################################################################
def plotGraphs(history): 
    
    #   Plot Accuracy Graph
    #-----------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("CNN Training vs Validation Accuracy")
    plt.savefig(os.path.join(OUTPUT_DIR,"TrainingAccuracy.png"))
    plt.legend()
    #plt.show()
    
    #  Plot Loss Graph 
    #-----------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("CNN Training vs Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR,"TrainingLoss.png"))

    #plt.show()
    
    
#########################################################################################################
#   Function Name    :  validateInputTestData
#   Description      :  Validates Input test Data files
#   Input Params     :  -   
#   Output Params    :  Positive and Negative Images
#   Author           :  Vaishali M Jorwekar
#   Date             :  29 May 2026
#########################################################################################################
def validateInputTestData():  
    
    #   Step 1  :   Check if Input dataset found
    #--------------------------------------
    checkFolder(POSITIVE_FOLDER)
    checkFolder(NEGATIVE_FOLDER)
    
    #   Step 2  :   Check if image folder contains valid images
    #--------------------------------------
    positiveImages=getValidImageFiles(POSITIVE_FOLDER)
    negativeImages=getValidImageFiles(NEGATIVE_FOLDER)

    print(f"Original Positive Images:{len(positiveImages)}")
    print(f"Original Negative Images:{len(negativeImages)}")

    #   Step 3  :   Check if image folder contains valid images
    #--------------------------------------
    if len(positiveImages) == 0 or len(negativeImages) == 0:
        print("ERROR: Positive or Negative folder contains no images.")
        exit()

    return positiveImages,negativeImages         
#########################################################################################################
#   Function Name    :  surfaceCrackDetection
#   Description      :  main function,manages calls to other functions
#   Input Params     :  -   
#   Output Params    :  -
#   Author           :  Vaishali M Jorwekar
#   Date             :  29 May 2026
#########################################################################################################
def surfaceCrackDetection():
    #   Validate Input data folder structure
    positiveImages,negativeImages=validateInputTestData()
    
    #   Create Industrial folder structure
    createIndustrialFolderStructure()
    
    #   Split and Copy images as "train,validation and test"
    splitDataSetCopyImages(POSITIVE_FOLDER, positiveImages, "Crack")
    splitDataSetCopyImages(NEGATIVE_FOLDER, negativeImages, "NoCrack")
    
    #   Define Train,test,Validation directories
    trainDir,validationDir,testDir=gettTrainTestValidationDir()
    
    #   Image preprocessing
    trainData,validationData,testData=imagePreprocessing(trainDir,validationDir,testDir)

    #   Build CNN model
    model=buildCNNModel()
    
    #   Train CNN Model  
    history=trainCNNModel(model,trainData,validationData)
    
    #   Plot graphs
    plotGraphs(history)
    
    #   Evaluate Model
    test_loss, test_accuracy=evaluateModel(model,testData)
    
    save_summary(test_accuracy,test_loss,len(history.history["loss"]),OUTPUT_DIR)

    #   Test Model
    testModel(model,testData)
###########################################################################################
#   Function        :   ensure_dir
#   Input Params    :   path(str)-directory path
#   Output Params   :   None
#   Description     :   Creates a directory if it does not exists
#   Author          :   Vaishali M Jorwekar
#   Date            :   3 Jun 2026
############################################################################################
def ensure_dir(path:str):
    os.makedirs(path,exist_ok=True)  
###########################################################################################
#   Function        :   parse_args
#   Input Params    :   None
#   Output Params   :   Parsed CLI arguments
#   Description     :   Defines command line arguments for training ,interference baselines
#   Author          :   Vaishali M Jorwekar
#   Date            :   3 Jun 2026
############################################################################################  
def parse_args():
    p=argparse.ArgumentParser(description="Surface Crack Detection Using CNN")
    p.add_argument('mode', choices=['train', 'test'], help="Execution mode: 'train' or 'test'")
    
    return p.parse_args()   
###########################################################################################
#   Function        :   testPretrainedModel
#   Input Params    :   testDir
#   Output Params   :   Testing model results
#   Description     :   Test pretrained model
#   Author          :   Vaishali M Jorwekar
#   Date            :   3 Jun 2026
############################################################################################  
def testPretrainedModel(testDir):  
    model=keras.models.load_model(BEST_MODEL) 
    testImageDataGen = ImageDataGenerator(rescale=1.0 / 255)
    testData = testImageDataGen.flow_from_directory(
                            testDir,
                            target_size=(IMAGE_SIZE, IMAGE_SIZE),
                            batch_size=BATCH_SIZE,
                            class_mode="binary",
                            shuffle=False
                        ) 
    testModel(model,testData)
#########################################################################################################
#   Function Name    :  main function 
#   Description      :  main function,manages calls to other functions
#   Input Params     :  -   
#   Output Params    :  -
#   Author           :  Vaishali M Jorwekar
#   Date             :  29 May 2026
#########################################################################################################
def main():
    ensure_dir(OUTPUT_DIR)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    print(BORDER)    
    print(BORDER)

    print("        Industrial Surface Crack Detection using CNN")
    print(BORDER)    
    print(BORDER)
    args=parse_args()
    if args.mode=="train":
        surfaceCrackDetection()
    elif args.mode == 'test':
        testDir = os.path.join(PROCESSED_DATASET, "test")  
        if not os.path.exists(testDir):  
            print(BORDER)
            print("Testing Directory does not exists.First train model") 
            exit()
        else:     
            testPretrainedModel(testDir)
            
#############################################################################################
#   Starter
##############################################################################################
if __name__=="__main__":
    main()
    