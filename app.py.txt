# app.py
import uvicorn
from fastapi import FastAPI

# Création de l'application FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Trading bot is running!"}

if __name__ == "__main__":
    # L'application écoute sur le port 10000
    uvicorn.run(app, host="0.0.0.0", port=10000)