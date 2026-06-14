## 📈 AI-Based Financial Time Series Forecasting using LSTM
*  Develope an AI-based stock price forecasting system using LSTM networks on the Reliance stock dataset.
*  This project focuses on predicting *'next day closing price'* of **Reliance stock** using **Deep learning(LSTM-Long Short Term Memory)**
Stock market data is Time series data where
*  Each value depends on previous value
*  Pattern Exsits
*  Sequential Learning required

### Overview
*  Implemented data preprocessing, time-series sequence generation
*  Used Deep Learning model-LSTM to predict future stock prices based on historical market trends
*  Evaluated model performance using RMSE, MAE, and graphical comparison of actual vs predicted values

---
## 📖 Dataset Information
<table>
      <thead>
        <tr>
          <th>Column</th>
          <th>Description</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Date</td>
          <td>Trading Date</td>
        </tr>
        <tr>
          <td>Open</td>
          <td>Price at market opening</td>
        </tr>
        <tr>
          <td>High</td>
          <td>Highest price of the day</td>
        </tr>
        <tr>
          <td>Low</td>
          <td>Lowest Price of the day</td>
        </tr>
        <tr>
          <td>Close</td>
          <td>Final price of the day</td>
        </tr> 
        <tr>
          <td>Volume</td>
          <td>Total shares traded</td>
        </tr>    
      </tbody>
</table>

## 🛠️ 💻 📚  Tech Stack
<table>
      <thead>
        <tr>
          <th>Technology</th>
          <th>Purpose</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Python</td>
          <td>Main Programming Language</td>
        </tr>
        <tr>
          <td>Keras</td>
          <td>LSTM Model</td>
        </tr>
        <tr>
          <td>Pandas</td>
          <td>Input data CSV read</td>
        </tr>
        <tr>
          <td>NumPy</td>
          <td>Numerical Operations</td>
        </tr>
        <tr>
          <td>Matplotlib</td>
          <td>Graph plot</td>
        </tr>
        <tr>
          <td>Scikit-learn</td>
          <td>Visual Result Display</td>
        </tr>
      </tbody>
</table>  

---
### 🚀 Key Features
*  In this project we are using only "Close" column of the dataset
*  Previous 10 days data is used for prdiction
*  LSTM is used to train the model  

###  📊 Visualisations
*  Actual Vs Predicted graph
*  Training Loss Graph
---
#### ✍️ Author
 Vaishali M. Jorwekar<br>
 Date	:  13 Jun 2026
  
 
