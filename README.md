# Speculator
This project aimed to create a new AI-based technical indicator to assist in trading decisions. For this purpose, historical data on stock prices and order books was curated. Then already existing technical indicators such as 20ema or l14 and h14 were computed and added to the dataset. Then neural network ensemble was trained with this data to predict the change in stock price. This model was deployed on the server using Flask API. 
- Reproducing results
To reproduce results download `html_clearverion` folder and open `index.html` file.
- Training
To train neural network ensemble download `net.py` and run `python net.py` in the corresponding folder.
- Final model and server code
Code for deploying model on server using Flask API is located in `server` folder. Final model is saved in a file located in `server/models` folder.
