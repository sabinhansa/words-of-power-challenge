import spacy
import requests
import time
import json
import joblib
import numpy as np

# Load spaCy model with disabled components for faster vector extraction.
nlp = spacy.load("en_core_web_md", disable=["tagger", "parser", "ner"])

host = "http://172.18.4.158:8000"
post_url = f"{host}/submit-word"
get_url = f"{host}/get-word"
status_url = f"{host}/status"
NUM_ROUNDS = 5

with open("response.json", "r") as f:
    player_words = json.load(f)

# Precompute and cache embeddings for player words.
player_embeddings = {}
for k, v in player_words.items():
    doc = nlp(v["text"])
    player_embeddings[k] = doc.vector

# Load the trained (fast) LightGBM cost predictor model.
model = joblib.load("cost_predictor_lgbm_fast.joblib")

def predict_cost(system_word):
    """
    Predicts the cost of a given system word using the LightGBM model.
    """
    doc = nlp(system_word)
    vec = doc.vector.reshape(1, -1)
    predicted_cost = model.predict(vec)[0]
    return max(predicted_cost, 0)

def choose_word(system_word, lambda_sim=1.0):
    """
    Given the system word, first predicts its cost. Then filters candidate words
    (from player_words) to only those whose cost is in the range:
         [predicted_cost, predicted_cost + 10].
    Among these candidates, the function computes cosine similarities in a vectorized
    manner and selects the candidate with the lowest similarity (i.e. the most semantically different).
    
    If no candidates fall within that cost range, it falls back to selecting the overall lowest-cost word.
    """
    predicted = predict_cost(system_word)
    lower_bound = predicted
    upper_bound = predicted + 10  # Increase cost range to predicted + 10

    # Filter candidates by cost range.
    valid = [(k, v) for k, v in player_words.items() if lower_bound <= v["cost"] <= upper_bound]
    if not valid:
        # Fallback: choose the overall lowest-cost word.
        return int(min(player_words, key=lambda k: player_words[k]["cost"]))
    
    # Build lists for valid candidate IDs and embeddings.
    valid_ids = []
    valid_embeddings = []
    for k, v in valid:
        valid_ids.append(k)
        valid_embeddings.append(player_embeddings[k])
    
    valid_embeddings = np.array(valid_embeddings)  # shape: (n_valid, vector_dim)
    
    # Compute system word vector (once) and its norm.
    system_vec = nlp(system_word).vector
    system_norm = np.linalg.norm(system_vec)
    
    # Compute norms of candidate embeddings.
    candidate_norms = np.linalg.norm(valid_embeddings, axis=1)  # shape: (n_valid,)
    
    # Compute cosine similarities in a vectorized way.
    cosine_similarities = np.dot(valid_embeddings, system_vec) / (system_norm * candidate_norms + 1e-8)
    
    # Select candidate with the lowest cosine similarity (most semantically different).
    best_index = np.argmin(cosine_similarities)
    best_id = valid_ids[best_index]
    return int(best_id)

def play_game(player_id):
    for round_id in range(1, NUM_ROUNDS + 1):
        round_info = {}
        # Poll for the correct round.
        while True:
            response = requests.get(get_url)
            data = response.json()
            if data["round"] == round_id:
                sys_word = data["word"]
                round_info["system_word"] = sys_word
                break
            time.sleep(0.1)
        
        if round_id > 1:
            status = requests.get(status_url)
            round_info["status"] = status.json()
        
        chosen_word_id = choose_word(sys_word, lambda_sim=1.0)
        chosen_word_text = player_words[str(chosen_word_id)]["text"]
        predicted = predict_cost(sys_word)
        round_info["predicted_cost"] = predicted
        round_info["chosen_word"] = chosen_word_text
        round_info["chosen_word_cost"] = player_words[str(chosen_word_id)]["cost"]
        
        payload = {"player_id": player_id, "word_id": chosen_word_id, "round_id": round_id}
        submit_resp = requests.post(post_url, json=payload)
        round_info["submission"] = submit_resp.json()
        
        print(f"\n--- Round {round_id} Overview ---")
        print(f"System word: {round_info['system_word']}")
        print(f"Predicted cost: {round_info['predicted_cost']:.2f}")
        print(f"Chosen word: {round_info['chosen_word']} (Cost: {round_info['chosen_word_cost']})")
        print(f"Submission response: {round_info['submission']}")
        if "status" in round_info:
            print(f"Status: {round_info['status']}")
        print("--- End of Round Overview ---\n")

if __name__ == "__main__":
    play_game("crocodilo bombardiero")
