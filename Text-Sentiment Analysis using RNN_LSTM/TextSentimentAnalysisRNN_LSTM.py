"""---------------------------------------------------------------------------------------------------------------
                Text-Sentiment Analysis using RNN/LSTM
                            Vaishali Jorwekar
-------------------------------------------------------------------------------------------------------------------
Problem statement:Build a Recurrent-Neural Network(RNN) with LSTM layers to predict sentiment
                 (Positive/Negative/Neutral) from text data such as movie reviews and social media posts
-------------------------------------------------------------------------------------------------------------------"""
#####################################################################################################
#   Imports
#####################################################################################################
import os,argparse
import pandas as pd
import keras
import joblib
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense,Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score

#####################################################################################################
#   Global configuration settings
#####################################################################################################
SEED=42
VAL_SPLIT=0.2
BORDER="-"*65
ARTIFACT_DIR="artifacts_TextSentiment"
BEST_MODEL=os.path.join(ARTIFACT_DIR,"sentiment_LSTM.h5")
FINAL_MODEL=os.path.join(ARTIFACT_DIR,"sentiment_LSTM_final.h5")
SENTIMENT_DATASET="IMDB Dataset.csv"
###########################################################################################
#   Function        :   readCSVFile
#   Input Params    :   dataSetFile
#   Output Params   :   Pandas data drame
#   Description     :   Load CSV data and return pandas data drame
#   Author          :   Vaishali M Jorwekar
#   Date            :   31 Jan 2026
############################################################################################
def readCSVFile(csvFileName)->pd.DataFrame:
    dFrame=pd.read_csv(csvFileName)
    print(BORDER)
    print(f"Data loaded successfully from file '{csvFileName}'")
    print(BORDER)
    print(f"File Data:\n{BORDER}\n{dFrame.head()}")
    print(f"Data Set Shape:{dFrame.shape}")
    print(f"Columns in data set:{dFrame.columns}")
   
    print(BORDER)
    return dFrame  
###########################################################################################
#   Function        :   ensure_dir
#   Input Params    :   path(str)-directory path
#   Output Params   :   None
#   Description     :   Creates a directory if it does not exists
#   Author          :   Vaishali M Jorwekar
#   Date            :   31 Jan 2026
############################################################################################
def ensure_dir(path:str):
    os.makedirs(path,exist_ok=True)
###########################################################################################
#   Function        :   parse_args
#   Input Params    :   None
#   Output Params   :   Parsed CLI arguments
#   Description     :   Defines command line arguments for training ,interference baselines
#   Author          :   Vaishali M Jorwekar
#   Date            :   31 Jan 2026
############################################################################################  
def parse_args():
    p=argparse.ArgumentParser(description="Text Sentiment Analysis using RNN/LSTM")
    p.add_argument("--train",action="store_true",help="Train and save artifacts")
    p.add_argument("--epochs",type=int,default=10,help="Epochs for training") 
    p.add_argument("--samples",type=int,default=9,help="Number of samples for inference grid")

    p.add_argument("--batch",type=int,default=128,help="Batch size for training") 
    p.add_argument("--lr",type=float,default=1e-3,help="Learning rate for Adam")
    p.add_argument("--infer",action="store_true",help="Generate an interence grid using saved model")
    p.add_argument("--baselines",action="store_true",help="Run Classical ML baslines")

    return p.parse_args()   
#############################################################################################
#   Function        :   load_data
#   Input Params    :   val_split(float)-Validation split ratio
#   Output Params   :   (train_data,val_data,test_data)->tuples
#   Description     :   Loads sentiment dataset ,normalize and split data into train,test sets
#   Author          :   Vaishali M Jorwekar
#   Date            :   31 Jan 2026
###############################################################################################
def load_data(val_split=0.2):
    dFrame=readCSVFile(SENTIMENT_DATASET)
    #   Convert Sentiment column to numerical values
    dFrame.replace({"sentiment":{"positive":1,"negative":0}},inplace=True)
    print(BORDER)

    print(f"Updated File Data:\n{BORDER}\n{dFrame.head()}")

    trainData,testData=train_test_split(dFrame,test_size=val_split,random_state=SEED)
    print(BORDER)
    print(f"Train Data set shape:{trainData.shape}")
    print(BORDER)
    print(f"Test Data set shape:{testData.shape}")
    
    dFrame.to_csv(os.path.join(ARTIFACT_DIR,"testData.csv"))
    return trainData,testData,dFrame
###########################################################################################
#   Function        :   preProcessData
#   Input Params    :   trainData,testData
#   Output Params   :   df(Data Frame)
#   Description     :   pre-process data
#   Author          :   Vaishali M Jorwekar
#   Date            :   31 Jan 2026
###########################################################################################
def preProcessData(trainData,testData,dFrame):  
    
    #   Tokenize data
    tokenizer=Tokenizer(num_words=5000)
    #dFrame['review']=dFrame['review'].astype(str)
    #tokenizer.fit_on_texts(trainData['review'].astype(str))
    tokenizer.fit_on_texts(trainData['review'])

    max_words = dFrame['review'].str.split().str.len().max()
    
    
     
    print(f"max Words len:{max_words}")
    totalWords=len(tokenizer.word_index)+1
    x_train=pad_sequences(tokenizer.texts_to_sequences(trainData['review']),maxlen=200)
    x_test=pad_sequences(tokenizer.texts_to_sequences(testData['review']),maxlen=200)

    
    y_train=trainData["sentiment"]
    y_test=testData["sentiment"]
    
    y_train = np.asarray(y_train).astype('int32')
    y_test = np.asarray(y_test).astype('int32')
    
    
    x_train,x_val,y_train,y_val=train_test_split(
                                x_train,y_train,
                                test_size=VAL_SPLIT,
                                random_state=SEED,
                                stratify=y_train)
    print(x_train.dtype)
    print(x_test.dtype)
    print(y_train.dtype)
    print(y_test.dtype)
    
    
    saveTrainedModel(tokenizer,"tokenizer")
    return (x_train,y_train),(x_test,y_test),(x_val,y_val)
###########################################################################################
#   Function        :   train_and_evaluate
#   Input Params    :   batch_size(int),epochs(int),lr(float)
#   Output Params   :   Trained model+saved artifacts
#   Description     :   Trains Sentiment Data,evaluates,saves curve,confusion matrix,
#                       misclassification, report
#   Author          :   Vaishali M Jorwekar
#   Date            :   31 Jan 2026
##########################################################################################
def train_and_evaluate(batch_size=128,epochs=10,lr=1e-3):
    ensure_dir(ARTIFACT_DIR) 
    trainData,testData,dFrame=load_data(val_split=VAL_SPLIT)
    (x_train,y_train),(x_test,y_test),(x_val,y_val)=preProcessData(trainData,testData,dFrame)

    print(x_train)
    #Build model
    model =build_lstm(lr)
    
    # Model Summary
    model.summary()
    
    callbacks=[
        #   saves best model during training.
        keras.callbacks.ModelCheckpoint(filepath=BEST_MODEL,monitor="val_accuracy",save_best_only=True,verbose=1),
        #   stops training automatically if the model's performance on the "val_loss" set stops improving,
        #   after '3' epochs
        keras.callbacks.EarlyStopping(patience=3,restore_best_weights=True,monitor="val_loss"),
        #   models performance reaches a plateau
        keras.callbacks.ReduceLROnPlateau(factor=0.5,patience=2,min_lr=1e-5,verbose=1)
    ]
    
    history=model.fit(
        x_train,y_train,
        batch_size=batch_size,
        callbacks=callbacks,
        epochs=epochs,
        validation_data=(x_val,y_val),
        verbose=2 )
    
    
    test_loss,test_acc=model.evaluate(x_test,y_test,verbose=0)
    
    print(f"[TEST] loss:{test_loss:.4f} |acc:{test_acc:.4f}")
    model.save(FINAL_MODEL)
    
    #Metrics and artifacts
    plot_training_curves(history,ARTIFACT_DIR)
    
    return history
###########################################################################################
#   Function        :   plot_training_curves
#   Input Params    :   history(keras.callbacks.History),out_dir(str)
#   Output Params   :   Saves Accuracy and Loss curve plots
#   Description     :   Plots and saves training/validation accuracy and loss curves
#   Author          :   Vaishali M Jorwekar
#   Date            :   31 Jan 2026
##########################################################################################
def plot_training_curves(history,out_dir=ARTIFACT_DIR):
    ensure_dir(out_dir)  
    #Accuracy    
    plt.figure(figsize=(7,5))
    plt.plot(history.history["accuracy"],label="train_acc")
    plt.plot(history.history["val_accuracy"],label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Vs Value Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"acuracyCurve.png"))
    plt.close()
    
    
    # Loss
    plt.figure(figsize=(7,5))
    plt.plot(history.history["loss"],label="train_loss")
    plt.plot(history.history["val_loss"],label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"Loss_Curve.png"))
    plt.close()
    
###########################################################################################
#   Function        :   build_cnn
#   Input Params    :   lr(float)-learning rate
#   Output Params   :   compiled Keras RNN/LSTM model
#   Description     :   Builds and compiles with RNN/LSTM 
#   Author          :   Vaishali M Jorwekar
#   Date            :   31 Jan 2026
##########################################################################################
def build_lstm(lr=1e-3)->keras.Model:  
    model=Sequential()
    model.add(keras.Input(shape=(200,)))
    model.add(Embedding(input_dim=5000,output_dim=128))
    model.add(LSTM(128,dropout=0.2,recurrent_dropout=0.2))
    model.add(Dense(1,activation="sigmoid"))
    
    
    opt=keras.optimizers.Adam(learning_rate=lr)
    
    model.compile(optimizer=opt,loss="binary_crossentropy",metrics=["accuracy"])
    
    return model 
###########################################################################################
#   Function        :   inference_grid
#   Input Params    :   n_samples(int),seed(int)
#   Output Params   :   Saved inference grid image
#   Description     :   Loads saved model,predicts on random test samples,
#                       saves prediction grid
#   Author          :   Vaishali M Jorwekar
#   Date            :   31 Jan 2026
###############################################################################
def inference_grid(n_samples=9,seed=7):
    if not os.path.exists(BEST_MODEL):
        print("Could not find {BEST_MODEL}. First train model with --train")
        return                
    testDataFile=os.path.join(ARTIFACT_DIR,"testData.csv") 
    tokenizer=loadTrainedModel("tokenizer")
    testDataFrame=readCSVFile(testDataFile)  
    testSeq=tokenizer.texts_to_sequences(testDataFrame['review'])
    padSeq=pad_sequences(testSeq,maxlen=200)
    model=keras.models.load_model(BEST_MODEL)
    predictedResult=model.predict(padSeq)
    
    testDataFrame["Predicted Emotion"]=predictedResult
    testDataFrame['Predicted Emotion'] = testDataFrame['Predicted Emotion'].apply(lambda x: 1 if x >= 0.5 else 0)

    print(testDataFrame.head())
    testDataFrame.to_csv(os.path.join(ARTIFACT_DIR,"Predicted.csv"))  
    
    testAcc=accuracy_score(testDataFrame["sentiment"],testDataFrame["Predicted Emotion"]) 
    
#####################################################################################################
#   Function name    :  loadTrainedModel
#   Input Params     :  path = MODEL_PATH
#   Output           :  model
#   Description      :  Load the trained model
#   Author           :  Vaishali M Jorwekar
#   Date             :  31 Jan 2026
#####################################################################################################
def loadTrainedModel(modelName):  
    path=modelName+".joblib"
    path=os.path.join(ARTIFACT_DIR,path)
    model = joblib.load(path)
    print(f"Model loaded from the path :{path}")
    return model     
#####################################################################################################
#   Function name    :  saveTrainedModel
#   Input Params     :  model,modelName
#   Output           :  -
#   Description      :  Save the trained model
#   Author          :   Vaishali M Jorwekar
#   Date            :   31 Jan 2026
#####################################################################################################
def saveTrainedModel(model,modelName):
    path=os.path.join(ARTIFACT_DIR,modelName+".joblib")
    joblib.dump(model,path)
    print(f"Model saved to path :{path}")       
############################################################################################
#   Function Name    :  textSentimentAnalysis
#   Description      :  main function 
#   Input Params     :  -   
#   Output Params    :  -
#   Author           :   Vaishali M Jorwekar
#   Date             :   31 Jan 2026
#############################################################################################
def textSentimentAnalysis():
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
#   Date             :   31 Jan 2026
#############################################################################################
def main():
    textSentimentAnalysis()
    
##############################################################################################
#   Starter
##############################################################################################
if __name__=="__main__":
    main()