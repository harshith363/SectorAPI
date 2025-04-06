from flask import Flask, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)
CORS(app)

tickers = ['XLY', 'XLP', 'XLE', 'XLK', 'XLV', 'XLF', 'XLI']

def get_sector_recommendations():
    data = yf.download(tickers, start='2010-01-01', end='2025-01-01', auto_adjust=False)['Adj Close']

    returns = data.pct_change().dropna()
    mean_returns = returns.mean(axis=1)
    labels = returns.apply(lambda x: (x > mean_returns).astype(int))

    rolling_features = pd.DataFrame()
    for ticker in tickers:
        rolling_features[f'{ticker}_5d'] = returns[ticker].rolling(5).mean()
        rolling_features[f'{ticker}_20d'] = returns[ticker].rolling(20).mean()

    rolling_features = rolling_features.dropna()
    labels = labels.loc[rolling_features.index]

    X_train, X_test, y_train, y_test = train_test_split(rolling_features, labels, test_size=0.2, random_state=42)

    sector_probabilities = {}
    sector_accuracies = {}

    for ticker in tickers:
        y_train_sector = y_train[ticker]
        y_test_sector = y_test[ticker]

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train_sector)

        sector_probabilities[ticker] = model.predict_proba(X_test)[:, 1].mean()
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test_sector, y_pred)
        sector_accuracies[ticker] = accuracy

    recommended_sectors = sorted(sector_probabilities.items(), key=lambda x: x[1], reverse=True)
    return [{"sector": sector, "probability": round(prob*100, 2)} for sector, prob in recommended_sectors]

@app.route('/recommend-sectors', methods=['GET'])
def recommend_sectors():
    try:
        recommendations = get_sector_recommendations()
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Remove debug mode for production
if __name__ == '__main__':
    app.run()
