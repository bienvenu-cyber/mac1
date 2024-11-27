import requests
import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from telegram import Bot
from concurrent.futures import ThreadPoolExecutor
import yfinance as yf
import os

# Initialisation des paramètres Telegram
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
bot = Bot(token=TELEGRAM_TOKEN)

# Liste des cryptomonnaies à surveiller
CRYPTO_LIST = ["BTC-USD", "ETH-USD", "ADA-USD"]  # Utiliser les tickers de yfinance

# Fichier de suivi des performances
PERFORMANCE_LOG = "trading_performance.csv"

# Fonction pour récupérer les données historiques avec yfinance
def fetch_crypto_data(crypto_id, period="1y"):
    data = yf.download(crypto_id, period=period)
    return data['Close'].values

# Fonction pour entraîner un modèle de machine learning (à améliorer)
def train_ml_model(data, target):
    # Si data est une liste ou un tableau 1D, on le reformate en 2D
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)  # Reformater en un tableau 2D

    # Division des données en ensemble d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

    # Modèle de régression logistique
    model = LogisticRegression()
    model.fit(X_train, y_train)

    return model

# Fonction pour calculer les indicateurs techniques
def calculate_indicators(prices):
    # Calculer des indicateurs plus complets (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
    sma_short = prices[-10:].mean()
    sma_long = prices[-20:].mean()
    # Exemple de plus d'indicateurs ici (simplifié pour le moment)
    return sma_short, sma_long

# Fonction pour analyser les signaux avec le modèle ML
def analyze_signals(prices, model):
    # Calculer les indicateurs
    indicators = calculate_indicators(prices)

    # Préparer les données pour le modèle
    features = np.array(indicators).reshape(1, -1)  # Assurer que les features sont en 2D
    prediction = model.predict(features)

    # Signal basé sur le modèle ML
    buy_signal = prediction[0] == 1

    # Calculer stop-loss et take-profit dynamiques (basés sur des indicateurs techniques)
    # Placeholder pour la logique du stop-loss et take-profit
    stop_loss = 0  # A remplir selon la stratégie
    take_profit = 0  # A remplir selon la stratégie

    return buy_signal, stop_loss, take_profit

# Fonction principale pour analyser une crypto
def analyze_crypto(crypto, model):
    prices = fetch_crypto_data(crypto)
    if prices is not None:
        buy_signal, stop_loss, take_profit = analyze_signals(prices, model)
        # Ici tu peux ajouter la logique pour envoyer les signaux via Telegram

# Fonction principale
def main():
    # Récupérer les données historiques de la cryptomonnaie
    data = fetch_crypto_data("BTC-USD", "5y")
    
    # Calculer les indicateurs (attention à la taille des features et targets)
    features = np.array([calculate_indicators(data[i-20:i]) for i in range(20, len(data))])
    targets = np.array([1 if data[i] > data[i-1] else 0 for i in range(20, len(data))])  # Ajuster l'index

    # Vérifier la taille des features et targets
    print(f"Size of features: {features.shape}")
    print(f"Size of targets: {targets.shape}")

    # Entraîner le modèle
    model = train_ml_model(features, targets)

    while True:
        with ThreadPoolExecutor() as executor:
            executor.map(lambda crypto: analyze_crypto(crypto, model), CRYPTO_LIST)
        time.sleep(300)

if __name__ == "__main__":
    main()