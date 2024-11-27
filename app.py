# app.py
from flask import Flask
from telegram import Bot
import time
from threading import Thread

# Initialisation des paramètres Telegram
TELEGRAM_TOKEN = "7402831359:AAHwrtwwqOhxsP4iajcx9-zGXev_DGDMlPY"
CHAT_ID = "1963161645"
bot = Bot(token=TELEGRAM_TOKEN)

app = Flask(__name__)

# Fonction pour envoyer des messages
def send_telegram_message(message):
    bot.send_message(chat_id=CHAT_ID, text=message)

# Fonction pour exécuter le bot en arrière-plan
def run_bot():
    while True:
        send_telegram_message("Bot is running...")
        time.sleep(300)

@app.route("/")
def home():
    return "Bot is running!"

if __name__ == "__main__":
    # Démarrer le bot dans un thread
    bot_thread = Thread(target=run_bot)
    bot_thread.daemon = True
    bot_thread.start()

    app.run(host="0.0.0.0", port=8000)