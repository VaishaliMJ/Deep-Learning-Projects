"""--------------------------------------------------------------------------------------------------
                    Apparel Classification Using Deep Learning
                    Student Name:   Vaishali Jorwekar
Problem Statement:  Apparel Classification using CNN(MNIST Dataset):
                    Built and trained a Convolutional Neural Network (CNN) model 
                    to classify Apparel               
-------------------------------------------------------------------------------------------------"""
import os,random
import argparse
import tensorflow as tf
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from typing import Tuple
from keras import layers
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#---------------------------------------------------------------------------------
#   Global configuration settings
#---------------------------------------------------------------------------------
SEED=42
ARTIFACT_DIR="artifacts_apparel"
BEST_MODEL=os.path.join(ARTIFACT_DIR,"apparel_cnn.h5")
FINAL_MODEL=os.path.join(ARTIFACT_DIR,"apparel_cnn_final.h5")
FASHION_CLASSES = ['t_shirt', 'trouser', 'pullover', 'dress', 'coat',
          'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boots']
#FASHION_CLASSES=[str(i) for i in range(len(FASHION_LABELS))]
###########################################################################################
#   Function        :   set_seed
#   Input Params    :   SEED(int)-random seed value
#   Output Params   :   None
#   Description     :   Sets seed for Python,NumPy,and TensorFlow to ensure reproducibilty
#   Author          :   Vaishali M Jorwekar
#   Date            :   25 Sep 2025
##########################################################################################
def set_seed(seed:int=SEED):
    #Activate Tensorflow deterministic behavior
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
#   Date            :   25 Sep 2025
############################################################################################
def ensure_dir(path:str):
    os.makedirs(path,exist_ok=True)
###########################################################################################
#   Function        :   parse_args
#   Input Params    :   None
#   Output Params   :   Parsed CLI arguments
#   Description     :   Defines command line arguments for training ,interference baselines
#   Author          :   Vaishali M Jorwekar
#   Date            :   25 Sep 2025
############################################################################################
def parse_args():
    p=argparse.ArgumentParser("Fashion-MNIST Case Study")
    p.add_argument("--train",action="store_true",help="Train the CNN and save artifacts")
    p.add_argument("--infer",action="store_true",help="Generate an inference grid using saved model")
    p.add_argument("--samples",type=int,default=9,help="Number of samples for inference grid")
    p.add_argument("--baselines",action="store_true",help="Run classical ML Models")
    p.add_argument("--epochs",type=int,default=15,help="Epochs for CNN Training")
    p.add_argument("--batch",type=int,default=128,help="Batch Size for CNN Training")
    p.add_argument("--lr",type=float,default=1e-3,help="Learning Rate for Adam")
    return p.parse_args()
###########################################################################################
#   Function        :   load_data
#   Input Params    :   val_split(float)-Validation split ratio
#   Output Params   :   (train_data,val_data,test_data)->tuples
#   Description     :   Loads MNIST dataset ,normalize and split
#                        data into train,val,test sets
#   Author          :   Vaishali M Jorwekar
#   Date            :   25 Sep 2025
###############################################################################
def load_data(val_split=0.1)->Tuple[Tuple[np.ndarray,np.ndarray],
                                    Tuple[np.ndarray,np.ndarray],
                                    Tuple[np.ndarray,np.ndarray]]:
    (x_train,y_train),(x_test,y_test)=keras.datasets.fashion_mnist.load_data()
    (x_train,x_val,y_train,y_val)=train_test_split(x_train,y_train,
                                                   test_size=val_split,
                                                   random_state=SEED,stratify=y_train)
    x_train=(x_train.astype("float32")/255.0)[...,None]
    x_val=(x_val.astype("float32")/255.0)[...,None]
    x_test=(x_test.astype("float32")/255.0)[...,None]
    return (x_train,y_train),(x_val,y_val),(x_test,y_test)
###########################################################################################
#   Function        :   build_cnn
#   Input Params    :   lr(float)-learning rate
#   Output Params   :   compiled Keras CNN model
#   Description     :   Builds and compiles with CNN with augumentation,
#                       Conv2D,Dropout,BatchNorm,Dense
#   Author          :   Vaishali M Jorwekar
#   Date            :   22 Sep 2025
##########################################################################################
def build_cnn(lr=1e-3)->keras.Model:
    data_augmentation=keras.Sequential(
        [
            layers.RandomTranslation(0.05,0.05,fill_mode="nearest"),
            layers.RandomRotation(0.05,fill_mode="nearest"),
            layers.RandomZoom(0.05,0.05,fill_mode="nearest"),

        ],
        name="augmentation"
    )
    inputs=keras.Input(shape=(28,28,1))
    x=data_augmentation(inputs)
    x=layers.Conv2D(32,3,padding="same",activation="relu")(x)
    x=layers.BatchNormalization()(x)
    x=layers.Conv2D(32,3,padding="same",activation="relu")(x)
    x=layers.MaxPooling2D()(x)

    x=layers.Conv2D(64,3,padding="same",activation="relu")(x)
    x=layers.BatchNormalization()(x)
    x=layers.Conv2D(64,3,padding="same",activation="relu")(x)
    x=layers.MaxPooling2D()(x)
    x=layers.Dropout(0.25)(x)

    x=layers.Flatten()(x)
    x=layers.Dense(128,activation="relu")(x)
    x=layers.BatchNormalization()(x)
    x=layers.Dropout(0.40)(x)

    outputs=layers.Dense(10,activation="softmax")(x)


    model=keras.Model(inputs,outputs,name="fashion_cnn")
    opt=keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt,loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    return model

###########################################################################################
#   Function        :   plot_training_curves
#   Input Params    :   history(keras.callbacks.History),out_dir(str)
#   Output Params   :   Saves Accuracy and Loss curve plots
#   Description     :   Plots and saves training/validation accuracy and loss curves
#   Author          :   Vaishali M Jorwekar
#   Date            :   22 Sep 2025
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
#   Description     :   Trains CNN on FASHION-MNIST,evaluates,saves curve,confusion matrix,
#                       misclassification, report
#   Author          :   Vaishali M Jorwekar
#   Date            :   21 Sep 2025
##########################################################################################
def train_and_evaluate(batch_size=128,epochs=10,lr=1e-3):
    ensure_dir(ARTIFACT_DIR)
    (x_train,y_train),(x_val,y_val),(x_test,y_test)=load_data(val_split=0.1)
    model=build_cnn(lr=lr)
    model.summary()
    callbacks=[
        keras.callbacks.ModelCheckpoint(BEST_MODEL,monitor="val_accuracy",save_best_only=True,verbose=1),
        keras.callbacks.EarlyStopping(patience=4,restore_best_weights=True,monitor="val_loss"),
        keras.callbacks.ReduceLROnPlateau(factor=0.5,patience=2,min_lr=1e-5,verbose=1)                               
    ]
    history=model.fit(
        x_train,y_train,
        validation_data=(x_val,y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2
    )
    test_loss,test_acc=model.evaluate(x_test,y_test,verbose=0)
    print(f"[Test]: loss:{test_loss:.4f} | acc:{test_acc:.4f}")

    model.save(FINAL_MODEL)
    plot_training_curves(history,ARTIFACT_DIR)

    y_pred=model.predict(x_test,batch_size=256).argmax(axis=1)
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
#   Date            :   23 Sep 2025
##########################################################################################
def save_summary(test_acc,test_loss,epochs,out_dir=ARTIFACT_DIR):
    ensure_dir(out_dir)
    with open(os.path.join(out_dir,"summary.txt"),"w") as f:
        f.write(
            "MNIST Digits CNN Summary\n"
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
#   Date            :   23 Sep 2025
##########################################################################################
def save_label_map(out_dir=ARTIFACT_DIR):

    ensure_dir(out_dir)
    with open(os.path.join(out_dir,"label_map.txt"),"w") as f:
        for i,name in enumerate(FASHION_CLASSES):
            f.write(f"{i}:{name}\n")

###########################################################################################
#   Function        :   plot_confusion_matrix
#   Input Params    :   y_true(np.ndarray),y_pred(np.ndarray),out_dir(str),normalize(bool)
#   Output Params   :   Saves Confusion Matrix Plot
#   Description     :   Generates and saves confusion matrix as Image
#   Author          :   Vaishali M Jorwekar
#   Date            :   22 Sep 2025
##########################################################################################
def plot_confusion_matrix(y_true,y_pred,out_dir=ARTIFACT_DIR,normalize=True):
    ensure_dir(out_dir)
    cm=confusion_matrix(y_true,y_pred)
    if normalize:
        cm=cm.astype("float")/(cm.sum(axis=1,keepdims=True))
    fig,ax=plt.subplots(figsize=(8.5,7))
    im=plt.imshow(cm,interpolation="nearest",cmap="Blues")
    ax.figure.colorbar(im,ax=ax)

    ax.set(
        xticks=(np.arange(len(FASHION_CLASSES))),
        yticks=(np.arange(len(FASHION_CLASSES))),
        xticklabels=FASHION_CLASSES,
        yticklabels=FASHION_CLASSES,
        xlabel="True Label",
        ylabel="Predicted Label",
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
#   Date            :   22 Sep 2025
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
#   Function        :   show_misclassifications
#   Input Params    :   x_true(np.ndarray),y_true(np.ndarray),
#                       y_pred(np.ndarray),limit(int),out_dir(str)
#   Output Params   :   Saves misclassification grid
#   Description     :   Displays and saves examples of misclassified test images
#   Author          :   Vaishali M Jorwekar
#   Date            :   23 Sep 2025
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
        plt.title(f"T:{FASHION_CLASSES[y_true[idx]]} \nP:{FASHION_CLASSES[y_pred[idx]]}",fontsize=9)
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
#   Date            :   22 Sep 2025
##########################################################################################
def save_classification_report(y_true,y_pred,out_dir=ARTIFACT_DIR):
    ensure_dir(out_dir)
    report=classification_report(y_true,y_pred,target_names=FASHION_CLASSES,digits=4)
    print(report)
    with open (os.path.join(out_dir,"classification_report.txt"),"w") as f:
        f.write(report)     
###########################################################################################
#   Function        :   inference_grid
#   Input Params    :   n_samples(int),seed(int)
#   Output Params   :   Saved inference grid image
#   Description     :   Loads saved model,predicts on random test samples,
#                       saves prediction grid
#   Author          :   Vaishali M Jorwekar
#   Date            :   22 Sep 2025
###############################################################################
def inference_grid(n_samples=9,seed=7):
    if not os.path.exists(BEST_MODEL):
        print("Could not find {BEST_MODEL}. First train model with --train")
        return         
    (_,_),(x_test,y_test)=keras.datasets.fashion_mnist.load_data()
    x_test=(x_test.astype("float32")/255.0)[...,None]
    model=keras.models.load_model(BEST_MODEL)

    rng=np.random.default_rng(seed)
    idx=rng.choice(len(x_test),size=n_samples,replace=False)

    imgs=x_test[idx]
    labs=y_test[idx]
    preds=model.predict(imgs,verbose=0).argmax(axis=1)

    cols=int(np.ceil(np.sqrt(n_samples)))
    rows=int(np.ceil(n_samples/cols))

    plt.figure(figsize=(2.8*cols,2.8*rows))
    for i in range(n_samples):
        plt.subplot(rows,cols,i+1)
        plt.imshow(imgs[i].squeeze(),cmap="gray")
        plt.title(f"P:{FASHION_CLASSES[preds[i]]}\nT:{FASHION_CLASSES[labs[i]]}",fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    ensure_dir(ARTIFACT_DIR)
    out_path=os.path.join(ARTIFACT_DIR,"inference_grid.png")
    plt.savefig(out_path)
    plt.close()
    print("Saved",out_path)  
###########################################################################################
#   Function        :   run_baselines
#   Input Params    :   None
#   Output Params   :   Prints accuracy of LogReg,LinearSVC,RandomForest
#   Description     :   Runs classical ML models on flattened MNIST digits
#   Author          :   Vaishali M Jorwekar
#   Date            :   22 Sep 2025
###########################################################################################
def run_baselines():
    x_train,y_train,x_test,y_test=load_flattened()
    
    logReg=Pipeline([
        ("scalar",StandardScaler(with_mean=True)),
        ("clf",LogisticRegression(max_iter=200,n_jobs=-1))
    ])
    logReg.fit(x_train,y_train)
    print(f"Logistic Regression  Acc: {accuracy_score(y_test,logReg.predict(x_test))}")


    svm=Pipeline([
        ("scalar",StandardScaler(with_mean=True)),
        ("clf",LinearSVC(C=1.0,dual=True,max_iter=5000))
    ])
    svm.fit(x_train,y_train)
    print(f"Linear SVC Acc:{accuracy_score(y_test,svm.predict(x_test))}")


    rf=RandomForestClassifier(n_estimators=200,max_depth=None,n_jobs=-1,random_state=SEED)
    rf.fit(x_train,y_train)
    print(f"Random Forest Acc:{accuracy_score(y_test,rf.predict(x_test))}")


###########################################################################################
#   Function        :   load_flattened
#   Input Params    :   None
#   Output Params   :   Flattened train and test arrays with labels
#   Description     :   Loads Fashion MNIST and flattens images for classical ML models
#   Author          :   Vaishali M Jorwekar
#   Date            :   26 Sep 2025
##########################################################################################
def load_flattened():
    (x_train,y_train),(x_test,y_test)=keras.datasets.fashion_mnist.load_data()
    x_train=x_train.reshape(len(x_train),-1).astype("float32")/255.0
    x_test=x_test.reshape(len(x_test),-1).astype("float32")/255.0
    return x_train,y_train,x_test,y_test
###############################################################################
#   Function        :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Entry point,Parse CLI arguments,
#                       runs train/infer/baselines accordingly
#   Author          :   Vaishali M Jorwekar
#   Date            :   21 Sep 2025
###############################################################################

def main():
    set_seed(SEED)
    ensure_dir(ARTIFACT_DIR)
    args=parse_args()
    did_anything=False
    if(args.train):
        train_and_evaluate(batch_size=args.batch,epochs=args.epochs,lr=args.lr)
        did_anything=True
    if args.infer:
        inference_grid(n_samples=args.samples)
        did_anything=True
    if args.baselines:
        run_baselines()
        did_anything=True

    if not did_anything:
        print("Nothing to do . Try:\n python3 Fashion_MNIST_CaseStudy.py --train")



###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()