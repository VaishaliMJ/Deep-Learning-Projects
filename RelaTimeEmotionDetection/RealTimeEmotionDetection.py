"""---------------------------------------------------------------------------------------------------------------
                Real Time Emotion Detection(CNN+OpenCV)
                            Vaishali Jorwekar
-------------------------------------------------------------------------------------------------------------------
Problem statement:Developed a real-time facial emotion recognition system using convolutional Neural Networks (CNN)
                  integrated with OpenCV for live video 
-------------------------------------------------------------------------------------------------------------------"""
#####################################################################################################
#   Imports
#####################################################################################################
import os
import argparse
import random
import numpy as np
import tensorflow as tf
import keras

#from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

#####################################################################################################
#   Global configuration settings
#####################################################################################################
SEED=42
ARTIFACT_DIR="artifacts_emotion"
BEST_MODEL=os.path.join(ARTIFACT_DIR,"emotion_cnn.h5")
FINAL_MODEL=os.path.join(ARTIFACT_DIR,"emotion_cnn_final.h5")
TRAIN_DATASET_DIR="dataset/train"
TEST_DATASET_DIR="dataset/test"
EMOTION_CLASSES=["angry","disgusted","fearful","happy","neutral","sad","suprised"]
###########################################################################################
#   Function        :   parse_args
#   Input Params    :   None
#   Output Params   :   Parsed CLI arguments
#   Description     :   Defines command line arguments for training ,interference baselines
#   Author          :   Vaishali M Jorwekar
#   Date            :   21 Jan 2026
############################################################################################  
def parse_args():
    p=argparse.ArgumentParser(description="Real time emoton detection Case Study")
    p.add_argument("--train",action="store_true",help="Train the CNN and save artifacts")
    p.add_argument("--epochs",type=int,default=10,help="Epochs for CNN training") 
    p.add_argument("--samples",type=int,default=9,help="Number of samples for inference grid")

    p.add_argument("--batch",type=int,default=128,help="Batch size for CNN training") 
    p.add_argument("--lr",type=float,default=1e-3,help="Learning rate for Adam")
    p.add_argument("--infer",action="store_true",help="Generate an interence grid using saved model")
    p.add_argument("--baselines",action="store_true",help="Run Classical ML baslines")

    return p.parse_args() 
###########################################################################################
#   Function        :   set_seed
#   Input Params    :   SEED(int)-random seed value
#   Output Params   :   None
#   Description     :   Sets seed for Python,NumPy,and TensorFlow to ensure reproducibilty
#   Author          :   Vaishali M Jorwekar
#   Date            :   21 Jan 2026
##########################################################################################
def set_seed(seed:int=SEED):
    os.environ["TF_DETERMINISTIC_OPS"]="1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
###########################################################################################
#   Function        :   ensure_dir
#   Input Params    :   path(str)-directory path
#   Output Params   :   None
#   Description     :   Creates a directory if it does not exists
#   Author          :   Vaishali M Jorwekar
#   Date            :   21 Jan 2026
############################################################################################
def ensure_dir(path:str): 
    os.makedirs(path,exist_ok=True) 
###########################################################################################
#   Function        :   processImage
#   Input Params    :   image,label
#   Output Params   :   None
#   Description     :   Normalize Image
#   Author          :   Vaishali M Jorwekar
#   Date            :   21 Jan 2026
############################################################################################
def processImage(image,label): 
    image=tf.cast(image/255.,tf.float32)
    return image,label     
#############################################################################################
#   Function        :   load_data
#   Input Params    :   val_split(float)-Validation split ratio
#   Output Params   :   (train_data,val_data,test_data)
#   Description     :   Loads dataset ,normalize and split data into train,val,test sets
#   Author          :   Vaishali M Jorwekar
#   Date            :   21 Jan 2026
###############################################################################################
def load_data():
    train_ds=image_dataset_from_directory(
        TRAIN_DATASET_DIR,
        labels='inferred',
        label_mode='categorical',
        color_mode='grayscale',
        batch_size=64,
        image_size=(48, 48)
    )
    validation_ds=image_dataset_from_directory(
        TEST_DATASET_DIR,
        labels='inferred',
        label_mode='categorical',
        color_mode='grayscale',
        batch_size=64,
        image_size=(48, 48)
    )
    train_ds=train_ds.map(processImage)
    validation_ds=validation_ds.map(processImage)

    return train_ds,validation_ds 
###########################################################################################
#   Function        :   build_cnn
#   Input Params    :   lr(float)-learning rate
#   Output Params   :   compiled Keras CNN model
#   Description     :   Builds and compiles with CNN with augumentation,
#                       Conv2D,Dropout,BatchNorm,Dense
#   Author          :   Vaishali M Jorwekar
#   Date            :   21 Jan 2026
##########################################################################################
def build_cnn(lr=1e-3)->keras.Model:
    data_augumentation=keras.Sequential(
        [ keras.layers.RandomTranslation(width_factor=0.05,height_factor=0.05,fill_mode="nearest"),
          keras.layers.RandomRotation(factor=0.05,fill_mode="nearest"),
          keras.layers.RandomZoom(height_factor=0.05,width_factor=0.05,fill_mode="nearest"),  
          #keras.layers.RandomFlip(mode="horizontal") 
        ],
        name="augumentation"
    )
    inputs=keras.Input(shape=(48,48,1))
    
    x=data_augumentation(inputs)
    
    x=keras.layers.Conv2D(filters=32,kernel_size=(3,3),padding="same",activation="relu")(x)
    x=keras.layers.BatchNormalization()(x)
    x=keras.layers.Conv2D(filters=32,kernel_size=(3,3),padding="same",activation="relu")(x)
    x=keras.layers.MaxPooling2D()(x)
    x=keras.layers.Dropout(0.25)(x)
    
    
    x=keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding="same",activation="relu")(x)
    x=keras.layers.BatchNormalization()(x)
    x=keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding="same",activation="relu")(x)
    x=keras.layers.MaxPooling2D()(x)
    x=keras.layers.Dropout(0.25)(x)
    
    x=keras.layers.Conv2D(filters=128,kernel_size=(3,3),padding="same",activation="relu")(x)
    x=keras.layers.BatchNormalization()(x)
    x=keras.layers.Conv2D(filters=128,kernel_size=(3,3),padding="same",activation="relu")(x)
    x=keras.layers.MaxPooling2D()(x)
    x=keras.layers.Dropout(0.25)(x)
    
    x=keras.layers.Conv2D(filters=256,kernel_size=(3,3),padding="same",activation="relu")(x)
    x=keras.layers.BatchNormalization()(x)
    x=keras.layers.Conv2D(filters=256,kernel_size=(3,3),padding="same",activation="relu")(x)
    x=keras.layers.MaxPooling2D()(x)
    x=keras.layers.Dropout(0.25)(x)
    
    """
    x=keras.layers.Conv2D(filters=512,kernel_size=(3,3),padding="same",activation="relu")(x)
    x=keras.layers.BatchNormalization()(x)
    x=keras.layers.Conv2D(filters=512,kernel_size=(3,3),padding="same",activation="relu")(x)
    x=keras.layers.MaxPooling2D()(x)
    x=keras.layers.Dropout(0.25)(x)"""
    
    x=keras.layers.Flatten()(x)
    x=keras.layers.Dense(128,activation="relu")(x)
    x=keras.layers.BatchNormalization()(x)
    x=keras.layers.Dropout(0.40)(x)
    
    outputs=keras.layers.Dense(7,activation="softmax")(x)
    
    model=keras.Model(inputs,outputs,name="emotion_cnn")
    opt=keras.optimizers.Adam(learning_rate=lr)
    
    model.compile(optimizer=opt,loss="categorical_crossentropy",metrics=["accuracy"])
    
    return model
###########################################################################################
#   Function        :   plot_training_curves
#   Input Params    :   history(keras.callbacks.History),out_dir(str)
#   Output Params   :   Saves Accuracy and Loss curve plots
#   Description     :   Plots and saves training/validation accuracy and loss curves
#   Author          :   Vaishali M Jorwekar
#   Date            :   21 Jan 2026
##########################################################################################
def plot_training_curves(history,out_dir=ARTIFACT_DIR):
    ensure_dir(out_dir)  
    #Accuracy
    plt.figure(figsize=(7,5))
    plt.plot(history.history["accuracy"],label="train_acc")
    plt.plot(history.history["val_accuracy"],label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Vs Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"acc_curve.png"))
    plt.close()
    
    
    #LOSS
    plt.figure(figsize=(7,5))
    plt.plot(history.history["loss"],label="train_loss")
    plt.plot(history.history["val_loss"],label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"loss_curve.png"))
    plt.close() 
###########################################################################################
#   Function        :   train_and_evaluate
#   Input Params    :   batch_size(int),epochs(int),lr(float)
#   Output Params   :   Trained model+saved artifacts
#   Description     :   Trains CNN on dataset,evaluates,saves curve,confusion matrix,
#                       misclassification, report
#   Author          :   Vaishali M Jorwekar
#   Date            :   21 Jan 2025
##########################################################################################
def train_and_evaluate(batch_size=128,epochs=10,lr=1e-3):
    ensure_dir(ARTIFACT_DIR)     
    train_ds,validation_ds =load_data()
    model =build_cnn(lr=lr)
    model.summary()
     # used during model training to automate tasks, prevent overfitting, and optimize performance. 
    callbacks=[
        #   saves best model during training.
        keras.callbacks.ModelCheckpoint(filepath=BEST_MODEL,monitor="val_accuracy",save_best_only=True,verbose=1),
        #   stops training automatically if the model's performance on the "val_loss" set stops improving,
        #   after '3' epochs
        keras.callbacks.EarlyStopping(patience=3,restore_best_weights=True,monitor="val_loss"),
        #   models performance reaches a plateau
        keras.callbacks.ReduceLROnPlateau(factor=0.5,patience=2,min_lr=1e-5,verbose=1)
    ]
    epochs=20
    history=model.fit(train_ds,
        validation_data=validation_ds,
        batch_size=batch_size,
        callbacks=callbacks,
        epochs=epochs,
        verbose=2  
    )
    
    model.save(FINAL_MODEL)

    #Metrics and artifacts
   #Metrics and artifacts
    plot_training_curves(history,ARTIFACT_DIR)
    x_test= np.concatenate([x for x, y in validation_ds], axis=0)
    y_test = np.concatenate([y for x, y in validation_ds], axis=0)
    #
    
    test_loss,test_acc=model.evaluate(x_test,y_test,verbose=0)
    
    
    y_pred=model.predict(validation_ds,batch_size=256).argmax(axis=1)
    #   Confusion matrix
    y_test = np.argmax(y_test, axis=1)
    plot_confusion_matrix(y_test,y_pred,ARTIFACT_DIR,normalize=True)
    save_classification_report(y_test,y_pred,ARTIFACT_DIR)
    show_misclassifications(x_test,y_test,y_pred,limit=25,out_dir=ARTIFACT_DIR)
    save_label_map(ARTIFACT_DIR)
    save_summary(test_acc,test_loss,len(history.history["loss"]),ARTIFACT_DIR)
###########################################################################################
#   Function        :   save_summary
#   Input Params    :   test_acc(float),test_loss(float),epochs(int),out_dir(str)
#   Output Params   :   Saves summary text file
#   Description     :   Saves final model performance summary and artifact list
#   Author          :   Vaishali M Jorwekar
#   Date            :   21 Jan 2026
##########################################################################################
def save_summary(test_acc,test_loss,epochs,out_dir=ARTIFACT_DIR):
    ensure_dir(out_dir)
    with open(os.path.join(out_dir,"summary.txt"),"w") as f:
        f.write(
            "Emotion Dataset CNN Summary\n"
            "================================================================\n"
            f"Test Accuracy :{test_acc:.4f}\n"
            f"Test Loss     :{test_loss:.4f}\n"
            f"Epochs    :{epochs}\n"
            f"Artifacts:acc.curve.png,loss_curve.png,confusion_matrix.png,\n"
            "           misscalssifications.png,digits_cnn.h5,digits_cnn_final.h5"


        )    
    
###########################################################################################
#   Function        :   save_label_map
#   Input Params    :   out_dir(str)
#   Output Params   :   Saves label mapping file
#   Description     :   Saves numeric-to-class name mapping (0-9) MNIST dataset
#   Author          :   Vaishali M Jorwekar
#   Date            :   21 Jan 2026
##########################################################################################
def save_label_map(out_dir=ARTIFACT_DIR):    
     with open(os.path.join(out_dir,"label_map.txt"),"w") as f:
        for i,name in enumerate(EMOTION_CLASSES):
            f.write(f"{i}:{name}\n")
                
###########################################################################################
#   Function        :   show_misclassifications
#   Input Params    :   x_true(np.ndarray),y_true(np.ndarray),
#                       y_pred(np.ndarray),limit(int),out_dir(str)
#   Output Params   :   Saves misclassification grid
#   Description     :   Displays and saves examples of misclassified test images
#   Author          :   Vaishali M Jorwekar
#   Date            :   21 Jan 2026
##########################################################################################
def show_misclassifications(x,y_true,y_pred,limit=25,out_dir=ARTIFACT_DIR):
    ensure_dir(out_dir)
    wrong=np.where(y_true!=y_pred)[0]
    if(len(wrong)==0):
        print("No Misclassifications !!!....")
        return 
    sel=wrong[:limit]
    cols=5
    rows=int(np.ceil(len(sel)/cols))
    plt.figure(figsize=(12,2.6*rows))
    for i,idx in enumerate(sel,1):
        img=x[idx].squeeze()
        plt.subplot(rows,cols,i)
        plt.imshow(img,cmap="gray")
        plt.title(f"T:{EMOTION_CLASSES[y_true[idx]]} \nP:{EMOTION_CLASSES[y_pred[idx]]}",fontsize=12)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"missclassification.png"))
    plt.close()     
###########################################################################################
#   Function        :   save_classification_report
#   Input Params    :   y_true(np.ndarray),y_pred(np.ndarray),out_dir(str)
#   Output Params   :   Saves classification report
#   Description     :   prints and saves classification report with precision,recall,F1 score
#   Author          :   Vaishali M Jorwekar
#   Date            :   21 Jan 2026
##########################################################################################
def save_classification_report(y_true,y_pred,out_dir=ARTIFACT_DIR):
    ensure_dir(out_dir)
    report=classification_report(y_true,y_pred,target_names=EMOTION_CLASSES,digits=4)
    print(report)
    with open (os.path.join(out_dir,"classification_report.txt"),"w") as f:
        f.write(report)
            
###########################################################################################
#   Function        :   plot_confusion_matrix
#   Input Params    :   y_true(np.ndarray),y_pred(np.ndarray),out_dir(str),normalize(bool)
#   Output Params   :   Saves Confusion Matrix Plot
#   Description     :   Generates and saves confusion matrix as Image
#   Author          :   Vaishali M Jorwekar
#   Date            :   21 Jan 2026
##########################################################################################
def plot_confusion_matrix(y_true,y_pred,out_dir=ARTIFACT_DIR,normalize=True):
    ensure_dir(out_dir)
    cm=confusion_matrix(y_true,y_pred)
    if normalize:
        cm=cm.astype("float")/cm.sum(axis=1,keepdims=True) 
    fig,ax=plt.subplots(figsize=(8.5,7))
    im=plt.imshow(cm,interpolation="nearest",cmap="Blues")
    ax.figure.colorbar(im,ax=ax)     
    ax.set(
        xticks=np.arange(len(EMOTION_CLASSES)),
        yticks=np.arange(len(EMOTION_CLASSES)),
        xticklabels=EMOTION_CLASSES,
        yticklabels=EMOTION_CLASSES,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="Confusion Matrix(Normalized)" if normalize else "Confusion Matrix"
    )  
    plt.setp(ax.get_xticklabels(),rotation=45,ha="right",rotation_mode="anchor")
    _annotate_confmat(ax,cm)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir,"confusion_matrix.png"))
    plt.close()  
###########################################################################################
#   Function        :   _annotate_confmat
#   Input Params    :   ax(matplotlib.axes.Axes),cm(np.ndarray),fmt(str)
#   Output Params   :   Annotated Confusion Matrix 
#   Description     :   Adds numeric annotations to confusion matrix heatmap
#   Author          :   Vaishali M Jorwekar
#   Date            :   21 Jan 2026
##########################################################################################
def _annotate_confmat(ax,cm,fmt="{:.2f}"):
    n,m=cm.shape   
    thresh=cm.max()/2.0
    for i in range (n):
        for j in range(m): 
         val=cm[i,j]
         color="white" if val > thresh else "black"
         txt=f"{val:.2f}" if isinstance(val,float) else f"{val:d}"
         ax.text(j,i,txt,ha="center",va="center",color=color,fontsize=8)      
###########################################################################################
#   Function        :   inference_grid
#   Input Params    :   n_samples(int),seed(int)
#   Output Params   :   Saved inference grid image
#   Description     :   Loads saved model,predicts on random test samples,
#                       saves prediction grid
#   Author          :   Vaishali M Jorwekar
#   Date            :   21 Jan 2026
###############################################################################
def inference_grid(n_samples=9,seed=7):
    if not os.path.exists(BEST_MODEL):
        print("Could not find {BEST_MODEL}. First train model with --train")
        return    
    validation_ds=image_dataset_from_directory(
        TEST_DATASET_DIR,
        label_mode='categorical',
        color_mode='grayscale',
        labels='inferred',
        batch_size=64,
        image_size=(48, 48),
        shuffle=True 
    )
    validation_ds = validation_ds.map(processImage)

    #   Load Best Model
    model=keras.models.load_model(BEST_MODEL)
    
    images = np.concatenate([x for x, y in validation_ds], axis=0)
    y_true_labels = np.concatenate([y for x, y in validation_ds], axis=0)
    y_true_labels = np.argmax(y_true_labels, axis=1)
   
    rng=np.random.default_rng(seed)
    idx=rng.choice(len(images),size=n_samples,replace=True)
    
    # Predict emotions
    predictions = model.predict(validation_ds)
    # Predicted classes
    predicted_indices = np.argmax(predictions, axis=1)

    
    imgs=images[idx]
    true_labels=y_true_labels[idx]
    predicted_labels=predicted_indices[idx]
    
    cols=int(np.ceil(np.sqrt(n_samples)))
    rows=int(np.ceil(n_samples/cols))   
    cols=int(np.ceil(np.sqrt(n_samples)))
    rows=int(np.ceil(n_samples/cols))   
    plt.figure(figsize=(2.8*cols,2.8*rows))
    
    for i in range(n_samples):
        plt.subplot(rows,cols,i+1)
        plt.imshow(imgs[i].squeeze(),cmap="gray")
        plt.title(f"P:{EMOTION_CLASSES[predicted_labels[i]]}\n T:{EMOTION_CLASSES[true_labels[i]]}",fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    ensure_dir(ARTIFACT_DIR)
    out_path=os.path.join(ARTIFACT_DIR,"inference_grid.png")
    plt.savefig(out_path)
    plt.close()
    print("Saved:",out_path)
    
    
    
#############################################################################################
#   Function Name    :  emotionDetection
#   Description      :  main function
#   Input Params     :  -   
#   Output Params    :  -
#   Author           :   Vaishali M Jorwekar
#   Date             :   21 Jan 2026
#############################################################################################
def emotionDetection():
    set_seed(SEED)
    ensure_dir(ARTIFACT_DIR)
    args=parse_args()
    did_anything=False
    if args.train:
        train_and_evaluate(batch_size=args.batch,epochs=args.epochs,lr=args.lr)
        did_anything=True
    if args.infer:
        inference_grid(n_samples=args.samples) 
        did_anything=True      
#############################################################################################
#   Function Name    :  main function 
#   Description      :  main function,manages calls to other functions
#   Input Params     :  -   
#   Output Params    :  -
#   Author           :   Vaishali M Jorwekar
#   Date             :   21 Jan 2026
#############################################################################################
def main():
    emotionDetection()
#############################################################################################
if __name__=="__main__":
    main()    