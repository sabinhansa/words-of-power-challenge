# Words of Power Challenge

**High-Risk, High-Reward NLP Game Strategy**

This repository presents a solution to the [Words of Power Hackathon Challenge](https://soleadify.notion.site/Hackathon-Challenge-Words-of-Power-1a52a4d999ed8021bb92dde896a630a5), a text-based competitive game where players must outsmart a system by selecting counter-words that "beat" the system-generated words. Each word has an associated cost, and the goal is to minimize the total cost across five rounds.

Our approach is **high-risk, high-reward**: the strategy leans heavily on selecting high-cost words that offer the **highest probability of winning**, minimizing penalties that would otherwise offset cheaper but riskier choices. This tradeoff ensures near-optimal performance in terms of total score.

---

## 🔍 Game Mechanics

In each of the 5 rounds:

- The system outputs a word (e.g., `"Volcano"`).
- The player must select a counter-word from a known vocabulary.
- Each word has:
  - A **monetary cost** (the more powerful or semantically rich, the higher the cost).
  - A **semantic effectiveness** (only known implicitly through data).
- The objective is to win each round by choosing a strong enough word while keeping your **total cost** (sum of all chosen word costs + penalties) as low as possible.

Penalty for losing a round is **significant**, so the strategy must strike a balance between picking powerful words and staying budget-aware.

---

## 🧠 Solution Overview

### Core Principle: *Better safe and expensive than cheap and wrong.*

This solution uses a **LightGBM regression model** trained to estimate how "effective" a counter-word is against a system word. Instead of random guessing or naive heuristics, the model uses real training data to predict the best counter-word for each input word.

- We score each available counter-word using model inference.
- We sort the words by score and choose the **top prediction**, regardless of cost.
- The assumption is that **winning consistently is cheaper overall** than taking a penalty, even if each individual word costs more.

---

## 🧱 File-by-File Breakdown

### `challenge_local.py`

- This is the main simulation driver.
- It mimics the behavior of the game:
  - Receives the system-generated word.
  - Queries the model for the best counter-word from the allowed list.
  - Simulates the result and tracks cumulative cost.
- Used extensively for local testing and strategy validation.

### `model.py`

- Defines the `Model` class that encapsulates:
  - Loading the LightGBM `.joblib` model.
  - Preprocessing candidate words.
  - Vectorizing features such as:
    - Word embeddings
    - TF-IDF scores
    - Semantic similarities
    - Custom-engineered features (length, word rarity, etc.)
  - Predicting a score (higher = more likely to win).
- The method `Model.predict(system_word, candidate_words)` returns the top counter-word.

### `cost_predictor_lgbm_fast.joblib`

- A pre-trained LightGBM regression model.
- Trained to estimate a proxy for how successful a word will be in the game.
- Input: Feature vector of (system_word, candidate_word) pair.
- Output: Predicted "power" score for the candidate.

### `train.csv`

- Dataset used for training the model.
- Each row contains:
  - A system word
  - A candidate counter-word
  - Associated metadata (embedding vectors, costs)
  - Labels indicating win/loss or effectiveness score
- Used to fit the LightGBM regressor during experimentation.

### `server.py`

- A lightweight Flask (or similar) API server.
- Hosts the model logic behind a simple HTTP interface.
- Allows external systems (e.g., evaluation scripts) to interact with the model via POST requests.

### `response.json`

- Example run of the model.
- Shows:
  - System words received
  - Counter-words selected
  - Their associated costs
  - Whether they won or lost
  - Final total cost

---

## ⚔️ Strategy: High-Risk, High-Reward

| Strengths                       | Weaknesses                      |
|--------------------------------|----------------------------------|
| Near-perfect win rate          | High per-word cost              |
| Avoids expensive penalties     | Not budget-optimized            |
| Fast decisions (pre-trained)   | Relies on model generalization  |

Rather than attempting to balance risk and reward dynamically, this model **maximizes confidence** at every step. The assumption: **penalties for failure outweigh potential savings from choosing cheaper words**.

---

## 🧪 Model Training (Summary)

- **Type**: LightGBM Regressor
- **Input Features**:
  - Word embeddings (e.g., fastText / GloVe)
  - One-hot encodings
  - Length, rarity, semantic distance, cosine similarity
- **Label**: Custom score based on past performance
- **Loss Function**: Mean Squared Error
- **Tuned for**: Maximum prediction accuracy over raw cost-efficiency

---

## 📌 Final Notes

This solution was crafted for a time-constrained hackathon, optimized for **maximum round success rate**. While it sacrifices budget optimization, it excels at consistently defeating the system with minimal logic overhead.

In future iterations, this could be extended with:
- Cost-aware decision layers (e.g. reinforcement learning)
- Dynamic budget balancing
- Uncertainty quantification
