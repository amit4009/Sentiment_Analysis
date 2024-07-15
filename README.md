# Sentiment Analysis with Neural Networks
<br>
<b>Project Overview</b>
<br>
This project is a comprehensive sentiment analysis application using neural networks. It processes IMDb movie reviews to determine whether the sentiment expressed in each review is positive or negative. The project employs various deep learning models, including a Simple Neural Network, a Convolutional Neural Network (CNN), and a Long Short-Term Memory (LSTM) network, to analyze the sentiments. 
<br>

# Dataset
<br>

The dataset used in this project is the IMDb Movie Reviews dataset, containing 50,000 reviews labeled as positive or negative. The dataset can be found on [Kaggle.](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
<br>

# Project Structure
<br>

 ├── IMDB_Dataset.csv <br>
 ├── glove.6B.100d.txt <br>
 ├── IMDb_Unseen_Reviews.csv <br>
 ├── IMDB_Review_sentiment_Analysis.ipynb<br>
 ├── lstm_model.h5<br>
 └── IMDb_Unseen_Reviews.csv (with predicted output) <br>
 
 <b>IMDB_Dataset.csv:</b> The main dataset containing 50,000 IMDb movie reviews.<br>
  <b>glove.6B.100d.txt: </b>Pre-trained GloVe word embeddings.<br>
  <b>IMDb_Unseen_Reviews.csv: </b>A sample dataset for making predictions with new, unseen reviews.<br>
  <b>IMDB_Review_sentiment_Analysis.ipynb: </b>Jupyter notebook containing the project code. <br> <b>lstm_model.h5:</b> Saved LSTM model with an accuracy of 85.6%.<br>
  <b>IMDb_Unseen_Predictions.csv:</b> Output predictions for new IMDb reviews.<br>

# Data Preprocessing
<br>
The data preprocessing steps include:
<br>
Removing special characters, numbers, and HTML tags from the reviews.<br>
Converting all text to lowercase.<br>
Removing stopwords using NLTK's stopwords list.<br>
Tokenizing and padding the sequences to ensure uniform input lengths for the models.<br>

# Word Embeddings
<br>
Word embeddings are used to convert textual data into numerical form. This project uses pre-trained GloVe embeddings to create an embedding matrix for the vocabulary in the dataset. The embeddings help capture semantic relationships between words.
<br>

![alt text](images/text_to_numbers.png)

![alt text](images/word_embedding.png)

# Model Architectures
<br>
Three different neural network architectures are used in this project:<br>

![alt text](images/architecture.png)

![alt text](images/Architecture_diagram_neuralNetwork.jpg)

# Training and Evaluation
<br>
The models are trained on the IMDb dataset with the following configurations:
<br>
Training-validation split: 80-20<br>
Batch size: 128<br>
Number of epochs: 20<br>
Loss function: Binary Crossentropy<br>
Optimizer: Adam<br>
The performance of each model is evaluated based on accuracy and loss on the test set.<br>

# Results
<br>
<b>Simple Neural Network:</b>
<br>
Test Accuracy: 75.45%<br>
Test Loss: 0.5867<br>

<b>Convolutional Neural Network (CNN):</b><br>

Test Accuracy: 84.54%<br>
Test Loss: 0.4195<br>

<b>Recurrent Neural Network (LSTM):</b><br>

Test Accuracy: 87.30%<br>
Test Loss: 0.3073<br>
The LSTM model achieves the highest accuracy and is used for making predictions on new data.<br>

# Predictions on New Data
<br>
The trained LSTM model is used to predict sentiments of new, unseen IMDb reviews. The predictions are saved in the c2_IMDb_Unseen_Predictions.csv file, with the predicted sentiments rounded to one decimal place.<br>

![alt text](images/Predicted_review.png)

# Web_app 
<br>

![alt text](images/web_app.png)

# Conclusion
<br>
This project demonstrates the effectiveness of different neural network architectures for sentiment analysis. The LSTM model, with its ability to capture long-term dependencies, outperforms the simple neural network and CNN models. The pre-trained GloVe embeddings enhance the model's understanding of semantic relationships between words.
<br>