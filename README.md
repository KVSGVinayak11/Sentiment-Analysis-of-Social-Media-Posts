# Sentiment-Analysis-of-Social-Media-Posts
# Twitter Sentiment Analysis

## Project Overview
This project performs sentiment analysis on Twitter data to classify tweets into sentiment categories such as Positive, Negative, and Neutral. Using natural language processing (NLP) techniques and deep learning models, the project aims to extract meaningful insights from social media text data.

The analysis pipeline includes data preprocessing, text cleaning, tokenization, model building with LSTM and Bidirectional layers, and evaluation of model performance.

## Dataset
The dataset consists of labeled tweets divided into training and testing sets:

- **Training data:** 74,682 tweets
- **Testing data:** 1,000 tweets

Each tweet is labeled with a sentiment category such as Positive, Negative, or Neutral. The dataset includes columns like tweet number, entity (e.g., Borderlands), sentiment label, and the tweet text.

## Technologies and Libraries Used
- Python 3.10
- Pandas, NumPy for data manipulation
- Matplotlib, Seaborn for visualization
- NLTK for text preprocessing (stopwords removal, tokenization)
- TensorFlow and Keras for deep learning model implementation
- Scikit-learn for model evaluation and preprocessing
- Pickle and Joblib for model serialization

## Methodology

1. **Data Loading:** Load training and testing datasets from CSV files.
2. **Data Preprocessing:** Clean tweets by removing stopwords, punctuation, and irrelevant tokens.
3. **Text Tokenization:** Convert text into sequences using Keras Tokenizer and pad sequences for uniform input length.
4. **Model Building:** Construct a Sequential deep learning model with Embedding, Bidirectional LSTM, Dropout, Batch Normalization, and Dense layers.
5. **Training:** Train the model with early stopping to prevent overfitting.
6. **Evaluation:** Assess model performance using accuracy, classification report, ROC curve, and precision-recall curve.
7. **Prediction:** Use the trained model to predict sentiments on new tweets.

## How to Use

1. Clone the repository:
2. Install required dependencies:
3. Run the Jupyter notebook `twitter-sentiment-analysis.ipynb` to execute the entire workflow.
4. Modify the notebook or scripts as needed to experiment with different models or datasets.

## File Structure

- `twitter-sentiment-analysis.ipynb` — Jupyter notebook containing the full analysis and model pipeline.
- `README.md` — Project documentation.

## Results
The model achieves high accuracy in classifying tweet sentiments, demonstrating the effectiveness of LSTM-based architectures for sentiment analysis tasks on Twitter data.

## Deployment
This project includes a sentiment analysis model deployed using Gradio for an interactive web interface. The deployment enables real-time sentiment predictions from user input, leveraging a pre-trained TensorFlow model and tokenizer. Gradio simplifies hosting, making the app accessible locally or via shareable links for easy testing and demonstration.


