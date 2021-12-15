# RedlegsStockPrediction
Made by Sumay Thakurdesai, Xinyi He, and Sayan Sisodiya. Uses a 9 day moving average and the average sentiment of tweets regarding a stock on a given day to predict the next day's stock price.

## A Note on Functionality
Our application "out of the box" can only predict stock prices for the companies Apple, Tesla, Meta (formerly Facebook; the stock symbol is still FB), and Amazon. This is due to time and space limitations for training and storing the machine learning models associated with each stock. More stocks can be added by running ```python3 main_model_training.py stock1 stock2 etc...``` with 1+ stocks. Note training does take a while, about 45 minutes per stock on Sumay's machine.

## Getting Started
You must have Python3 installed to use our application. To build our application, clone the GitHub repository into your local environment. Then, from the ```final-project-redlegs``` directory in the terminal, run the command ```pip install -r requirements.txt``` to install all of our application's dependencies.

To run the application once this is complete, run ```python3 predict_prices.py``` file from the terminal with a stock symbol as the sole argument. For instance, if you wanted to get the prediction for Apple's stock, you would run the command ```python3 predict_prices.py AAPL```. 
