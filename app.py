from flask import Flask, jsonify
from flask_cors import CORS 
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


app = Flask(__name__)
CORS(app)

tickers = ['XLY', 'XLP', 'XLE', 'XLK', 'XLV', 'XLF', 'XLI']

def get_sector_recommendations():
    data = yf.download(tickers, start='2010-01-01', end='2025-01-01', auto_adjust=False)['Adj Close']

# Calculate daily returns
    returns = data.pct_change().dropna()
    mean_returns = returns.mean(axis=1)

    # Create labels: 1 if sector return > mean return, else 0
    labels = returns.apply(lambda x: (x > mean_returns).astype(int))

    # Create rolling average features (5-day and 20-day moving averages)
    rolling_features = pd.DataFrame()
    for ticker in tickers:
        rolling_features[f'{ticker}_5d'] = returns[ticker].rolling(5).mean()
        rolling_features[f'{ticker}_20d'] = returns[ticker].rolling(20).mean()

    # Drop NaN values and align labels
    rolling_features = rolling_features.dropna()
    labels = labels.loc[rolling_features.index]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(rolling_features, labels, test_size=0.2, random_state=42)

    # Predict probabilities for each sector individually
    sector_probabilities = {}
    sector_accuracies = {}

    for i, ticker in enumerate(tickers):
        # Extract labels for this specific sector
        y_train_sector = y_train[ticker]
        y_test_sector = y_test[ticker]

        # Train a separate model for each sector
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train_sector)

        # Get the probability of outperformance (class=1) for this sector
        sector_probabilities[ticker] = model.predict_proba(X_test)[:, 1].mean()

        # Calculate accuracy for this sector
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test_sector, y_pred)
        sector_accuracies[ticker] = accuracy

    # Sort sectors based on highest probability of outperformance
    recommended_sectors = sorted(sector_probabilities.items(), key=lambda x: x[1], reverse=True)

    return [{"sector": sector, "probability": round(prob*100, 2)} for sector, prob in recommended_sectors]

@app.route('/recommend-sectors', methods=['GET'])
def recommend_sectors():
    recommendations = get_sector_recommendations()
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
