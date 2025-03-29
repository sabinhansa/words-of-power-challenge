from flask import Flask, jsonify, request
import random
import json

app = Flask(__name__)

with open("response.json", "r") as f:
    player_words = json.load(f)

system_word_requirements = {
    "Flood": 2,
    "Tornado": 3,
    "Hammer": 1,
    "Pandemic": 4,
    "Fire": 2,
    "Ice": 1,
    "Storm": 3,
    "Earthquake": 4,
    "Volcano": 5,
    "Hurricane": 4
}

system_words = list(system_word_requirements.keys())
round_number = 1

@app.route("/words", methods=["GET"])
def words():
    return jsonify({"player_words": player_words, "system_word_requirements": system_word_requirements})

@app.route("/get-word", methods=["GET"])
def get_word():
    global round_number
    word = random.choice(system_words)
    return jsonify({"word": word, "round": round_number})

@app.route("/submit-word", methods=["POST"])
def submit_word():
    data = request.get_json()
    return jsonify({"result": "ok", "data": data})

@app.route("/status", methods=["GET"])
def status():
    return jsonify({"status": "Round completed"})

@app.route("/next-round", methods=["POST"])
def next_round():
    global round_number
    round_number += 1
    return jsonify({"round": round_number})

if __name__ == "__main__":
    app.run(debug=True)
