from flask import Flask, request, jsonify
import random

app = Flask(__name__)

# Sample words to use for testing.
words = [
    "Dust",
    "Flame",
    "Shadow",
    "Thunder",
    "Ice"
]

# Dummy status data for testing.
status_data = {
    "round": 5,
    "game_over": True,
    "p1_total_cost": 100,
    "p2_total_cost": 90,
    "p1_word": "Flame",
    "p2_word": "Thunder",
    "p1_won": True,
    "p2_won": False
}

@app.route("/get-word", methods=["GET"])
def get_word():
    # Randomly pick a round between 1 and 5 and a random word.
    round_number = random.randint(1, 5)
    word = random.choice(words)
    return jsonify({"round": round_number, "word": word})

@app.route("/status", methods=["GET"])
def status():
    return jsonify(status_data)

@app.route("/submit-word", methods=["POST"])
def submit_word():
    data = request.get_json()
    # For testing, simply echo the data back with a status.
    return jsonify({"status": "ok", "received": data})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
