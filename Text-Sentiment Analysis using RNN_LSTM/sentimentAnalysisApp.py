"""---------------------------------------------------------------------------------------------------------------
                        Text Sentiment Analysis App
                    Sentiment Analysis with RNN/LSTM
                            Vaishali Jorwekar
-------------------------------------------------------------------------------------------------------------------
Problem statement:Text-Sentiment Analysis using RNN/LSTM
-------------------------------------------------------------------------------------------------------------------"""
#####################################################################################################
#   Imports
#####################################################################################################
import os,joblib,re
import keras
import streamlit as st
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#####################################################################################################
#   Constants and file names
#####################################################################################################
BORDER="-"*65
ARTIFACT_DIR="artifacts_TextSentiment"
BEST_MODEL=os.path.join(ARTIFACT_DIR,"sentiment_LSTM.h5")

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
#########################################################################################################
#   Function Name    :  predictSentiment
#   Description      :  This function predicts sentiment of movie review
#   Input Params     :  review 
#   Output Params    :  -
#   Author           :  Vaishali M Jorwekar
#   Date             :  31 Jan 2026
#########################################################################################################
def predictSentiment(review):
    tokenizer=loadTrainedModel("tokenizer")

    testSeq=tokenizer.texts_to_sequences([review])
    padSeq=pad_sequences(testSeq,maxlen=200)
    model=keras.models.load_model(BEST_MODEL)
    predictedResult=model.predict(padSeq)
    sentiment=1 if predictedResult[0][0] >=0.5 else 0
    return sentiment
 
#####################################################################################################
#   Function name    :  analyseSentiment
#   Input Params     :  sentimentPredicted
#   Output           :  model
#   Description      :  Convert into String format
#   Author           :  Vaishali M Jorwekar
#   Date             :  31 Jan 2026
#####################################################################################################
def analyseSentiment(sentimentPredicted): 
    if sentimentPredicted == 1 :
        label="Positive 😊" 
        color="green"
    elif sentimentPredicted==0 :
        label="Negative 😔" 
        color="red"
    return label,color         
#########################################################################################################
#   Function Name    :  main function 
#   Description      :  main function,manages calls to other functions
#   Input Params     :  -   
#   Output Params    :  -
#   Author           :  Vaishali M Jorwekar
#   Date             :  31 Jan 2026
#########################################################################################################
def main():
    st.title("🚀 Text Sentiment Analyzer : Movie Review(😊😔)")

    if not os.path.exists(BEST_MODEL):
        print("Could not find {BEST_MODEL}. First train model with --train")
        return                
    testDataFile=os.path.join(ARTIFACT_DIR,"testData.csv") 
    user_input=st.text_area("Enter movie review to analyse sentiment",value="")
        
    predictedSentiment = predictSentiment(user_input)
    if st.button("🚀 Predict Sentiment"):
        
        label = ""
        color=""
        # negative :0, positive : 1
        st.write("**Movie Review Text Sentiment**")   
        label,color=analyseSentiment(predictedSentiment)
        st.write(f"** :{color}[{label}]")
            
        st.info(f"**Review (Sentiment):** ")
    else:
        st.warning("Please enter some review first.")
    st.sidebar.markdown("### Model Details\n")
    st.sidebar.write("- **Text Sentiment analysis with:** LSTM")
    
#########################################################################################################
#   Starter
#########################################################################################################
if __name__=="__main__":
    main()