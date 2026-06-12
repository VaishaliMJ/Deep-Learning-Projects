"""---------------------------------------------------------------------------------------------------------------
                Project Name    :   AI-Based Financial Time Series Forecasting using LSTM(Reliance)
                Author          :  Vaishali Jorwekar
                Date            :  8 Jun 2026 
-------------------------------------------------------------------------------------------------------------------
Problem statement   : Develope an AI-based stock price forecasting system using LSTM networks
                      on the Reliance stock dataset.
-------------------------------------------------------------------------------------------------------------------"""
#####################################################################################################
#   Imports
#####################################################################################################

#os.environ["ARROW_DISABLE_PTHREAD_MUTEX"] = "1"

import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
from keras.callbacks import EarlyStopping

BORDER="="*80
DATASET_FILE="ProjectData/marvellous_reliance_stock_sample.csv"
OUTPUT_DIR="ProjectOutput"
TIME_STEP=10
SPLIT_SIZE=0.80
#############################################################################################
#   Function Name    :   printTitle
#   Description      :   This function prints header/title
#   Input Params     :   -   
#   Output Params    :   Header printed
#   Author           :   Vaishali M Jorwekar
#   Date             :   8 Jun 2026
#############################################################################################    
def printTitle(title):
    print(f"\n{BORDER}")
    print(title.center(80))
    print(BORDER)

#############################################################################################
#   Function Name    :   displayTable
#   Description      :   This function prints data in table format
#   Input Params     :   -   
#   Output Params    :   Data in table format
#   Author           :   Vaishali M Jorwekar
#   Date             :   8 Jun 2026
#############################################################################################    
def displayTable(title, dataframe, rows=10):
    printTitle(title)
    print(dataframe.head(rows).to_string(index=False))

#############################################################################################
#   Function Name    :   loadStockCSV
#   Description      :   This function loads Reliance Stock sample CSV file
#   Input Params     :   -   
#   Output Params    :   Loaded CSV file   
#   Author           :   Vaishali M Jorwekar
#   Date             :   8 Jun 2026
#############################################################################################
def loadStockCSV():
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(
            f"Dataset file not found. Please keep '{DATASET_FILE}' in the folder."
        )
    dataFile=pd.read_csv(DATASET_FILE)
    displayTable("STEP 1 : ORIGINAL DATASET VALUES",dataFile,rows=15)
    print(f"\nDataset Shape :    {dataFile.shape}")
    print(f"Total Rows      :    {dataFile.shape[0]}")
    print(f"Total Columns   :    {dataFile.shape[1]}")
    print(f"Column Names    :    {list(dataFile.columns)}")

    print(f"\nDataset Data Types:")
    print(dataFile.dtypes)
    return dataFile
#############################################################################################
#   Function Name    :   convertDateColumn
#   Description      :   This function converts date column into Date DateType
#   Input Params     :   dataFile 
#   Output Params    :   Converted Date Column 
#   Author           :   Vaishali M Jorwekar
#   Date             :   8 Jun 2026
#############################################################################################
def convertDateColumn(dataFile):
    printTitle("STEP 2 : DATE CONVERSION AND SORTING")
    print(f"Before Conversion, Date Column Type:{dataFile["Date"].dtype}")
    dataFile["Date"] = pd.to_datetime(dataFile["Date"])
    dataFile = dataFile.sort_values("Date").reset_index(drop=True)

    print(f"After conversion, Date column type :{dataFile["Date"].dtype}")
    print(f"\nFirst Date in Dataset:{dataFile["Date"].iloc[0].date()}")
    print(f"Last Date in Dataset :{dataFile["Date"].iloc[-1].date()}")
#############################################################################################
#   Function Name    :   extractClosePrice
#   Description      :   This function extracts 'Close' price column
#   Input Params     :   dataFile 
#   Output Params    :   Extracted Column Value
#   Author           :   Vaishali M Jorwekar
#   Date             :   9 Jun 2026
#############################################################################################
def extractClosePrice(dataFile):
    printTitle("STEP 3 : EXTRACT CLOSE PRICE FOR FORECASTING")
    stockClosePrices = dataFile[["Close"]].values

    print("Close price values are extracted as a NumPy array.")
    print(f"Shape of Stock Close Prices:{stockClosePrices.shape}")
    print("\nFirst 10 Close Prices")
    print(BORDER)
    print(f"Day     Close Price")
    print(BORDER)

    for i in range(min(10, len(stockClosePrices))):
        print(f"{i+1:02d}       {stockClosePrices[i][0]}")
    return stockClosePrices    
#############################################################################################
#   Function Name    :   minMaxScaling
#   Description      :   Applied Min-Max scaling on "Close" price
#   Input Params     :   stockClosePrices 
#   Output Params    :   Scaled values
#   Author           :   Vaishali M Jorwekar
#   Date             :   9 Jun 2026
#############################################################################################
def minMaxScaling(stockClosePrices):
    printTitle("STEP 4 : MIN-MAX SCALING WITH MANUAL CALCULATION")

    minPrice=stockClosePrices.min()
    maxPrice=stockClosePrices.max()
    
    minMaxDiff=maxPrice-minPrice
    print(f"Minimum Close Price:{minPrice}")
    print(f"Maximum Close Price:{maxPrice}")
    
    print("\nFormula of Min-Max Scaling:")
    print("Scaled Value = (Original Value - Minimum Value) / (Maximum Value - Minimum Value)")

    print("\nManual scaling calculation for first 5 records:")
    for i in range(min(5, len(stockClosePrices))):
        original = stockClosePrices[i][0]
        scaled_manual = (original - minPrice) / (minMaxDiff)
        print(
            f"Record {i+1}: ({original} - {minPrice}) / "
            f"({maxPrice} - {minPrice}) = {scaled_manual:.6f}"
        )
    # Actual scaling using sklearn
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close = scaler.fit_transform(stockClosePrices)

    print("\nFirst 10 scaled values produced by MinMaxScaler:")
    print(BORDER)
    print(f"Day \t Original \t  Scaled ")
    print(BORDER)
    for i in range(min(10, len(scaled_close))):
        print(f" {i+1:02d} \t {stockClosePrices[i][0]}  \t {scaled_close[i][0]:.6f}")
    return scaled_close,scaler
#############################################################################################
#   Function Name    :   createSequences
#   Description      :   Previous 10 days are used to predict the next day
#   Input Params     :   scaledStockClosePrices,scaler
#   Output Params    :   Scaled values
#   Author           :   Vaishali M Jorwekar
#   Date             :   9 Jun 2026
#############################################################################################
def createSequences(scaledStockClosePrices,scaler):
    printTitle("STEP 5 : TIME SERIES SEQUENCE CREATION")
    X = []
    y = []

     
    for i in range(TIME_STEP, len(scaledStockClosePrices)):
        previous_days = scaledStockClosePrices[i-TIME_STEP:i, 0]
        next_day = scaledStockClosePrices[i, 0]

        X.append(previous_days)
        y.append(next_day)
    X=np.array(X)
    y=np.array(y)
    print(f"\nTotal sequences created:{len(X)}")
    print(f"Shape of X before reshape:{X.shape}")
    print(f"Shape of y:{y.shape}")
    
    print("\nFirst 3 sequences with scaled values:")
    for seqNum in range(min(3, len(X))):
        print(f"\nSequence Number:{seqNum + 1}")
        print("Input X values:")
        for dayIndex in range(0,TIME_STEP):
            print(f"  Previous Day {dayIndex+1:02d}: {X[seqNum][dayIndex]:.6f}")
        print(f"Output y value: {y[seqNum]:.6f}")
    print("\nSame first sequence in original price values:")
    firstSequenceOriginal = scaler.inverse_transform(X[0].reshape(-1, 1))
    firstOutputOriginal = scaler.inverse_transform([[y[0]]])
    for i, value in enumerate(firstSequenceOriginal):
        print(f"Input Day {i+1:02d}: {value[0]:.2f}")
    print("Output Next Day:", round(firstOutputOriginal[0][0], 2))


    return X,y
#############################################################################################
#   Function Name    :   reshapeStockClose
#   Description      :   Reshape Stock Close Prices
#   Input Params     :   X
#   Output Params    :   Reshaped Stock 'Close' Prices
#   Author           :   Vaishali M Jorwekar
#   Date             :   9 Jun 2026
#############################################################################################
def reshapeStockClose(X):
    printTitle("STEP 6 : RESHAPE DATA FOR LSTM INPUT")

    print("LSTM expects 3D input:")
    print("[Number of Samples, Number of Time Steps, Number of Features]")
    print("\nIn our project:")
    print("Samples    = Total number of sequences")
    print("Time Steps = 10 previous days")
    print("Features   = 1 because we use only Close price")

    X = X.reshape(X.shape[0], X.shape[1], 1)

    print("\nShape of X after reshape:", X.shape)
    print("Shape meaning:", X.shape[0], "samples,", X.shape[1], "time steps,", X.shape[2], "feature")
    return X
#############################################################################################
#   Function Name    :   internalWorkingInfo
#   Description      :   Prints internal information of LSTM Gates and Calculations
#   Input Params     :   None
#   Output Params    :   None
#   Author           :   Vaishali M Jorwekar
#   Date             :   9 Jun 2026
#############################################################################################
def internalWorkingInfo():
    printTitle("STEP 7 : LSTM INTERNAL WORKING CONCEPT")

    print("LSTM is an improved version of RNN.")
    print("It is useful for sequential data like stock prices, text, weather, sales, etc.")
    print("\nLSTM maintains two important states:")
    print("1. Hidden State h_t  : Short-term output information")
    print("2. Cell State C_t    : Long-term memory information")

    print("\nLSTM contains four main calculations:")
    print("1. Forget Gate     : Decides what old memory should be forgotten")
    print("2. Input Gate      : Decides what new information should be accepted")
    print("3. Candidate Memory: Creates possible new memory")
    print("4. Output Gate     : Decides current output hidden state")

    print("\nMathematical Formulas:")
    print("Forget Gate      f_t = sigmoid(W_f * [h_(t-1), x_t] + b_f)")
    print("Input Gate       i_t = sigmoid(W_i * [h_(t-1), x_t] + b_i)")
    print("Candidate Memory C~t = tanh(W_c * [h_(t-1), x_t] + b_c)")
    print("Cell State       C_t = f_t * C_(t-1) + i_t * C~t")
    print("Output Gate      o_t = sigmoid(W_o * [h_(t-1), x_t] + b_o)")
    print("Hidden State     h_t = o_t * tanh(C_t)")
#############################################################################################
#   Function Name    :   splitDataSet
#   Description      :   Split Data into Train and Test Data set
#   Input Params     :   X,y
#   Output Params    :   trainX,trainY,testX,testY
#   Author           :   Vaishali M Jorwekar
#   Date             :   9 Jun 2026
#############################################################################################
def splitDataSet(X,y):
    printTitle("STEP 8 : TRAIN TEST SPLIT")

    train_size = int(len(X) * SPLIT_SIZE)

    trainX = X[:train_size]
    testX = X[train_size:]
    trainY = y[:train_size]
    testY = y[train_size:]

    print(f"Total Records after sequence creation:{len(X)}")
    print(f"Training Records 80%:{len(trainX)}")
    print(f"Testing Records 20% :{len(testX)}")

    print(f"\nTraining X shape:{trainX.shape}")
    print(f"Training y shape:{trainY.shape}")
    print(f"Testing X shape :{testX.shape}")
    print(f"Testing y shape :{testY.shape}")
        
    return trainX,trainY,testX,testY
#############################################################################################
#   Function Name    :   buildLSTMModel
#   Description      :   Build LSTM Model
#   Input Params     :   None
#   Output Params    :   model
#   Author           :   Vaishali M Jorwekar
#   Date             :   9 Jun 2026
#############################################################################################
def buildLSTMModel():
    printTitle("STEP 9 : BUILD LSTM MODEL")

    model = Sequential()

    # return_sequences=True is required because another LSTM layer is connected after this layer.
    model.add(LSTM(units=50, return_sequences=True, input_shape=(TIME_STEP, 1)))
    model.add(Dropout(0.2))

    # Last LSTM layer returns only final output, so return_sequences=False.
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))

    # Dense layer performs final decision-making / regression mapping.
    model.add(Dense(units=25, activation="relu"))

    # Output layer has one neuron because we predict one value: next Close price.
    model.add(Dense(units=1))

    model.compile(optimizer="adam", loss="mean_squared_error")

    print("Model is compiled using:")
    print("Optimizer : Adam")
    print("Loss      : Mean Squared Error")
    print("\nModel Summary:")
    model.summary()

    print("\nLayer Explanation:")
    print("LSTM Layer 1  : Reads sequence of 10 days and returns sequence output")
    print("Dropout 1     : Reduces overfitting by ignoring 20% neurons randomly")
    print("LSTM Layer 2  : Learns final temporal pattern")
    print("Dropout 2     : Again reduces overfitting")
    print("Dense 25      : Learns nonlinear combination of LSTM output")
    print("Dense 1       : Predicts next day scaled close price")

    return model
#############################################################################################
#   Function Name    :   trainModel
#   Description      :   Train LSTM Model
#   Input Params     :   model,trainX,trainY
#   Output Params    :   None
#   Author           :   Vaishali M Jorwekar
#   Date             :   9 Jun 2026
#############################################################################################
def trainModel(model,trainX,trainY):
    printTitle("STEP 10 : MODEL TRAINING")

    print("During training, model compares predicted value with actual value.")
    print("Then it updates internal weights using backpropagation through time.")
    print("\nEpoch means one complete pass over training data.")
    print("Batch size means number of samples processed before weight update.")
    
    early_stop = EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True
            )

    history = model.fit(
        trainX,
        trainY,
        epochs=60,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    print("\nTraining completed.")
    print("Total epochs actually executed:", len(history.history["loss"]))
    print("Final training loss  :", history.history["loss"][-1])
    print("Final validation loss:", history.history["val_loss"][-1])
    
    return history
#############################################################################################
#   Function Name    :   testModel
#   Description      :   Test LSTM Model
#   Input Params     :   model,testX,testY
#   Output Params    :   Predicted Results
#   Author           :   Vaishali M Jorwekar
#   Date             :   9 Jun 2026
#############################################################################################
def testModel(model,testX,testY):
    printTitle("STEP 11 : MODEL TESTING")
    predictedScaledResults=model.predict(testX)
    
    print("Predicted values are currently scaled between 0 and 1.")
    print("\nFirst 10 scaled predictions vs actual scaled values:")
    print(BORDER)
    print("Record \t\tActual Scaled Prices \tPredicted Scaled Results")
    print(BORDER)
    for i in range(min(10, len(predictedScaledResults))):
        print(
         f"{i+1:02d} \t\t {testY[i]:.6f}  \t\t  {predictedScaledResults[i][0]:.6f}"
        )
    return predictedScaledResults  
#############################################################################################
#   Function Name    :   inverseScaling
#   Description      :   Inverse Scaling,Convert back values to Original
#   Input Params     :   model,testX,testY
#   Output Params    :   Predicted Results
#   Author           :   Vaishali M Jorwekar
#   Date             :   9 Jun 2026
#############################################################################################
def inverseScaling(scaler,predictedScaledResults,testY):  
    printTitle("STEP 12 : CONVERT SCALED VALUES BACK TO ORIGINAL PRICE")

    predictedPrices = scaler.inverse_transform(predictedScaledResults)
    actualPrices = scaler.inverse_transform(testY.reshape(-1, 1))
    print("\nFirst 10 predictions in original value:")
    print(BORDER)
    print("Record\tActual Price\tPredicted Price\t  Difference ")
    print(BORDER)
    for i in range(min(10, len(predictedPrices))):
        print(
            f" {i+1:02d}\t{actualPrices[i][0]:.2f}, "
            f"\t{predictedPrices[i][0]:.2f}, "
            f"\t {actualPrices[i][0] - predictedPrices[i][0]:.2f}"
         )
    return predictedPrices,actualPrices 
#############################################################################################
#   Function Name    :   errorCalculations
#   Description      :   Error Calculations
#   Input Params     :   predictedPrices,actualPrices
#   Output Params    :   Predicted Results
#   Author           :   Vaishali M Jorwekar
#   Date             :   9 Jun 2026
#############################################################################################
def errorCalculations(predictedPrices,actualPrices): 
    printTitle("STEP 13 : MODEL EVALUATION WITH CALCULATIONS")
 
    fileData="" 
    meanAbsError=mean_absolute_error(actualPrices,predictedPrices)
    meanSquaredError=mean_squared_error(actualPrices,predictedPrices)
    rmse = np.sqrt(meanSquaredError) 
    
    fileData=fileData+f"{BORDER}"+"\n\t\t\tModel Error Details\n"
    fileData=fileData+f"{BORDER}\n"
    fileData=fileData+f"Mean Absolute Error :   {round(meanAbsError, 2)}\n"
    fileData=fileData+f"Mean Squared Error  :   {round(meanSquaredError, 2)}\n"
    fileData=fileData+f"Root MSE            :   {round(rmse, 2)}\n"
    print(fileData)
    ensure_dir(OUTPUT_DIR)
    outDir=os.path.join(OUTPUT_DIR,"MeanSquaredError.txt")
    with open(outDir,"w") as fileTxt:
         fileTxt.write(str(fileData))
    
#####################################################################################################
#   Function Name   :   ensure_dir
#   Input Params    :   path of directory
#   Output Params   :   None
#   Description     :   Check and create ARTIFACT_DIR if does not exists
#   Author          :   Vaishali M. Jorwekar
#   Date            :   9 Jun 2026
#####################################################################################################
def ensure_dir(path:str):
    os.makedirs(path,exist_ok=True) 
#############################################################################################
#   Function Name    :   plotActualVsPredictedGraph
#   Description      :   Plot Loss Graphs
#   Input Params     :   predictedPrices,actualPrices
#   Output Params    :   Graphs
#   Author           :   Vaishali M Jorwekar
#   Date             :   9 Jun 2026
#############################################################################################
def plotActualVsPredictedGraph(predictedPrices,actualPrices):   
    
    printTitle("STEP 14 : ACTUAL VS PREDICTED GRAPH")

    plt.figure(figsize=(12, 6))
    plt.plot(actualPrices, label="Actual Reliance Close Price")
    plt.plot(predictedPrices, label="Predicted Reliance Close Price")
    plt.title("Reliance Stock Price Forecasting using LSTM")
    plt.xlabel("Test Record Number")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True)
    ensure_dir(OUTPUT_DIR)
    plt.savefig(os.path.join(OUTPUT_DIR,"Actual_vs_Predicted.png"))
    plt.show()

    print("Graph saved as Actual_vs_Predicted.png")
    
#############################################################################################
#   Function Name    :   plotLossGraph
#   Description      :   Plot Loss Graphs
#   Input Params     :   predictedPrices,actualPrices
#   Output Params    :   Graphs
#   Author           :   Vaishali M Jorwekar
#   Date             :   9 Jun 2026
#############################################################################################
def plotLossGraph(history):   
    printTitle("STEP 15 : TRAINING LOSS GRAPH")

    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Marvellous LSTM Training Loss")
    plt.xlabel("Epoch Number")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    ensure_dir(OUTPUT_DIR)

    plt.savefig(os.path.join(OUTPUT_DIR,"Training_Loss_Graph.png"))
    plt.show()

    print("Graph saved as Training_Loss_Graph.png")
#############################################################################################
#   Function Name    :   nextDayPrediction
#   Description      :   Next Day Prediction
#   Input Params     :   scaledStockClosePrices,scaler
#   Output Params    :   Graphs
#   Author           :   Vaishali M Jorwekar
#   Date             :   9 Jun 2026
#############################################################################################
def nextDayPrediction(scaledStockClosePrices,scaler,model):
    prevTenDayPrices=scaledStockClosePrices[-TIME_STEP:]
    print("Last 10 days used for next-day prediction:")
    last_10_original = scaler.inverse_transform(prevTenDayPrices)
    print(BORDER)
    print(" Day\t\tOriginal\t\tScaled")
    print(BORDER)
    for i in range(TIME_STEP):
        print(
            f" {i+1:02d}\t\t {last_10_original[i][0]:.2f}, "
            f"\t\t{prevTenDayPrices[i][0]:.6f}"
        )
    last_10_days = prevTenDayPrices.reshape(1, TIME_STEP, 1)
    nextDayScaled = model.predict(last_10_days)
    nextDayPrice = scaler.inverse_transform(nextDayScaled)

    print("\nNext Day Prediction:")
    print("Predicted Scaled Value:", round(float(nextDayScaled[0][0]), 6))
    print("Predicted Original Close Price:", round(float(nextDayPrice[0][0]), 2))
#############################################################################################
#   Function Name    :   saveModelAndPriceValues
#   Description      :   Save Model,Predicted And Actual values
#   Input Params     :   model,actualPrices,predictedPrices
#   Output Params    :   Graphs
#   Author           :   Vaishali M Jorwekar
#   Date             :   9 Jun 2026
#############################################################################################
def saveModelAndPriceValues(model,actualPrices,predictedPrices):
    printTitle("STEP 17 : SAVE MODEL AND PREDICTION OUTPUT")
    ensure_dir(OUTPUT_DIR)
    outDirPath=os.path.join(OUTPUT_DIR,"Reliance_Price_Prediction.h5")
    model.save(outDirPath)
    
    outputDf = pd.DataFrame({
    "Actual_Close_Price": actualPrices.flatten(),
    "Predicted_Close_Price": predictedPrices.flatten(),
    "Difference": (actualPrices.flatten() - predictedPrices.flatten())
    })
    
    outputDf.to_csv(os.path.join(OUTPUT_DIR,"OutputPredicted.csv"))
    print("Model Saved as 'Reliance_Price_Prediction.h5'")  
      
#############################################################################################
#   Function Name    :  main function 
#   Description      :  main function,manages calls to other functions
#   Input Params     :  -   
#   Output Params    :  -
#   Author           :   Vaishali M Jorwekar
#   Date             :   8 Jun 2026
#############################################################################################
def main():
    #   Step 1  :    Load CSV
    dataFile=loadStockCSV()

    #   Step 2  :   Date Conversion
    convertDateColumn(dataFile)  
    
    #   Step 3  :   Extract "Close Price" for forcasting
    stockClosePrices=extractClosePrice(dataFile)

    #   Step 4  :   Min-Max Scaling
    scaledStockClosePrices,scaler=minMaxScaling(stockClosePrices)
    
    #   Step 5  :   Create Sequences based on previous inputs
    X,y=createSequences(scaledStockClosePrices,scaler)
    
    #   Step 6  :   Reshape Input for LSTM
    X=reshapeStockClose(X)
    
    #   Step 7  :   Internal Working concept of LSTM
    internalWorkingInfo()
    
    #   Step 8  :   Split Data Set into Train Test Set
    trainX,trainY,testX,testY=splitDataSet(X,y)
    
    #   Step 9  :   Build LSTM Model
    model=buildLSTMModel()
    
    #   Step 10  :   Train Model
    history=trainModel(model,trainX,trainY)
    
    #   Step 11  :  Test Model
    predictedScaledResults=testModel(model,testX,testY)
    
    #   Step 12 :   Inverse Scaling
    predictedPrices,actualPrices=inverseScaling(scaler,predictedScaledResults,testY)
    
    #   Step 13 :   Error Calculations
    errorCalculations(predictedPrices,actualPrices)
    
    #   Step 14 :   Plot Graphs
    plotActualVsPredictedGraph(predictedPrices,actualPrices)
    
    #   Step 15 :   Plot Loss Graph
    plotLossGraph(history)
    
    #   Step 16 :   Next Day Prediction
    nextDayPrediction(scaledStockClosePrices,scaler,model)
    
    #   Step 17 :   Save Models and Price Predicted And Actual Values 
    saveModelAndPriceValues(model,actualPrices,predictedPrices)
##############################################################################################
#   Starter
##############################################################################################
if __name__=="__main__":
    main()