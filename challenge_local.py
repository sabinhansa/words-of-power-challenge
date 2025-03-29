import warnings
warnings.simplefilter("ignore")

import spacy
import requests
import time
import json
import joblib
import numpy as np

nlp = spacy.load("en_core_web_md", disable=["tagger", "parser", "ner"])
host = "http://localhost:8000"
post_url = f"{host}/submit-word"
get_url = f"{host}/get-word"
status_url = f"{host}/status"
NUM_ROUNDS = 5

with open("response.json", "r") as f:
    player_words = json.load(f)

player_embeddings = {}
for k, v in player_words.items():
    doc = nlp(v["text"])
    player_embeddings[k] = doc.vector

model = joblib.load("cost_predictor_lgbm_fast.joblib")

def predict_cost(system_word):
    doc = nlp(system_word)
    vec = doc.vector.reshape(1, -1)
    predicted_cost = model.predict(vec)[0]
    return max(predicted_cost, 0)

def choosen_word(system_word):
    predicted = predict_cost(system_word)
    lower_bound = predicted
    upper_bound = predicted + 10
    valid = [(k, v) for k, v in player_words.items() if lower_bound <= v["cost"] <= upper_bound]
    if not valid:
        return int(min(player_words, key=lambda k: player_words[k]["cost"]))
    valid_ids = []
    valid_embeddings = []
    for k, v in valid:
        valid_ids.append(k)
        valid_embeddings.append(player_embeddings[k])
    valid_embeddings = np.array(valid_embeddings)
    system_vec = nlp(system_word).vector
    system_norm = np.linalg.norm(system_vec)
    candidate_norms = np.linalg.norm(valid_embeddings, axis=1)
    cosine_similarities = np.dot(valid_embeddings, system_vec) / (system_norm * candidate_norms + 1e-8)
    best_index = np.argmin(cosine_similarities)
    best_id = valid_ids[best_index]
    return int(best_id)

def what_beats(system_word):
    return choosen_word(system_word)

def play_game(player_id):

    for round_id in range(1, NUM_ROUNDS+1):
        round_num = -1
        while round_num != round_id:
            response = requests.get(get_url)
            print(response.json())
            sys_word = response.json()['word']
            round_num = response.json()['round']

            time.sleep(1)

        if round_id > 1:
            status = requests.get(status_url)
            print(status.json())

        choosen_word = what_beats(sys_word)
        data = {"player_id": player_id, "word_id": choosen_word, "round_id": round_id}
        response = requests.post(post_url, json=data)
        print(response.json())

if __name__ == "__main__":
    play_game("64PkL80DLf")
