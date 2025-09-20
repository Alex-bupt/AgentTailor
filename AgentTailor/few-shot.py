import os
import math
import json
import pandas as pd
import random
from typing import List, Dict, Tuple
from collections import defaultdict
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)
from langchain.chains import LLMChain

# ----------------------
# 1. Deepseek API Configuration
# ----------------------
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"
os.environ["OPENAI_API_KEY"] = "your API Key"  # Please replace with your Deepseek API Key

# ----------------------
# 2. File Paths
# ----------------------
RATINGS_FILE = "./ml-1m new/ratings_cleaned.txt"
USER_RECS_CSV = "./ml-1m new/user_recommendations.csv"
USER_LABELS_CSV = "./ml-1m new/user_labels.csv"
MOVIES_JSON = "./ml-1m new/movies_filled.json"
MOVIE_POPULARITY_CSV = "./ml-1m new/movie_popularity.csv"

model_name = "deepseek-chat"

# ----------------------
# 3. Load Data from Files
# ----------------------
# Load user ratings and split into training/testing sets
user_ratings_all: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
with open(RATINGS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("::")
        if len(parts) < 4:
            continue
        uid_str, mid_str, rating_str, _ = parts
        uid, mid = int(uid_str), int(mid_str)
        rating = float(rating_str)
        user_ratings_all[uid].append((mid, rating))

user_train: Dict[int, List[Tuple[int, float]]] = {}
user_test: Dict[int, Tuple[int, float]] = {}
for uid, hist in user_ratings_all.items():
    if hist:
        user_train[uid] = hist[:-1]
        user_test[uid] = hist[-1]
    else:
        user_train[uid] = []
        user_test[uid] = (None, 0.0)

# Load precomputed Top-20 recommendation results
user_recs_df = pd.read_csv(USER_RECS_CSV)
user_recommendations: Dict[int, List[int]] = {
    int(row['UserID']): [int(row[f'Rec{i + 1}']) for i in range(20)]
    for _, row in user_recs_df.iterrows()
}

# Load test labels
labels_df = pd.read_csv(USER_LABELS_CSV)
user_labels: Dict[int, int] = {
    int(row['UserID']): int(row['TestMovieID'])
    for _, row in labels_df.iterrows()
}

# Load movie metadata
with open(MOVIES_JSON, 'r', encoding='utf-8') as f:
    movies_data = json.load(f)
movie_info: Dict[int, Dict] = {int(item['MovieID']): item for item in movies_data}

# Load movie popularity data
popularity_df = pd.read_csv(MOVIE_POPULARITY_CSV)
movie_popularity: Dict[int, str] = {
    int(row['MovieID']): row['Popularity']
    for _, row in popularity_df.iterrows()
}

# ----------------------
# 4. LLM Client Setup
# ----------------------
llm = ChatOpenAI(model_name=model_name, temperature=1.0)

# ----------------------
# 5. Define the Single-Call Few-Shot Prompt
# ----------------------
few_shot_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are an expert movie recommender. Your task is to analyze a user's viewing history and rerank a given list of candidate movies. "
        "The final output must be a single JSON array containing only the movie IDs, ordered from most to least recommended. "
        "Do not include any other text or explanation."
    ),
    # Example 1: User likes Action/Sci-Fi
    HumanMessagePromptTemplate.from_template(
        "User History:\n"
        "- Movie: 'Terminator 2: Judgment Day' (ID: 589), Genres: ['Action', 'Sci-Fi'], User Rating: 5.0\n"
        "- Movie: 'The Matrix' (ID: 2571), Genres: ['Action', 'Sci-Fi', 'Thriller'], User Rating: 5.0\n"
        "- Movie: 'Star Wars: Episode IV - A New Hope' (ID: 260), Genres: ['Action', 'Adventure', 'Sci-Fi'], User Rating: 4.0\n\n"
        "Candidate Movies to Rerank:\n"
        "- Movie: 'Pretty Woman' (ID: 924), Genres: ['Comedy', 'Romance']\n"
        "- Movie: 'Blade Runner' (ID: 1246), Genres: ['Film-Noir', 'Sci-Fi']\n"
        "- Movie: 'Die Hard' (ID: 1036), Genres: ['Action', 'Thriller']\n"
        "- Movie: 'The Fugitive' (ID: 457), Genres: ['Action', 'Thriller']"
    ),
    AIMessagePromptTemplate.from_template("[1036, 457, 1246, 924]"),
    # Example 2: User likes Classic Drama/Crime
    HumanMessagePromptTemplate.from_template(
        "User History:\n"
        "- Movie: 'The Godfather' (ID: 858), Genres: ['Action', 'Crime', 'Drama'], User Rating: 5.0\n"
        "- Movie: 'Pulp Fiction' (ID: 296), Genres: ['Crime', 'Drama'], User Rating: 5.0\n"
        "- Movie: 'Casablanca' (ID: 913), Genres: ['Drama', 'Romance', 'War'], User Rating: 4.0\n\n"
        "Candidate Movies to Rerank:\n"
        "- Movie: 'GoodFellas' (ID: 780), Genres: ['Crime', 'Drama']\n"
        "- Movie: 'Toy Story' (ID: 1), Genres: ['Animation', 'Children''s', 'Comedy']\n"
        "- Movie: 'The Shawshank Redemption' (ID: 318), Genres: ['Drama']\n"
        "- Movie: 'Braveheart' (ID: 110), Genres: ['Action', 'Drama', 'War']"
    ),
    AIMessagePromptTemplate.from_template("[780, 318, 110, 1]"),
    # Actual Task
    HumanMessagePromptTemplate.from_template(
        "User History:\n{history}\n\nCandidate Movies to Rerank:\n{candidates}"
    ),
])

baseline_chain = LLMChain(llm=llm, prompt=few_shot_prompt, output_key="reranked_list")

# ----------------------
# 6. Helper Functions to Build Inputs for the Baseline
# ----------------------
def format_history_for_prompt(history_list: List[Dict]) -> str:
    """Formats user history into a string for the prompt."""
    lines = []
    for item in history_list:
        lines.append(
            f"- Movie: '{item['title']}' (ID: {item['id']}), Genres: {item['genres']}, User Rating: {item['user_rating']}"
        )
    return "\n".join(lines)

def format_candidates_for_prompt(candidate_list: List[Dict]) -> str:
    """Formats candidate movies into a string for the prompt."""
    lines = []
    for item in candidate_list:
        lines.append(
            f"- Movie: '{item['title']}' (ID: {item['id']}), Genres: {item['genres']}"
        )
    return "\n".join(lines)

def get_user_history(user_id: int) -> List[Dict]:
    """Reused from original script."""
    history = []
    recent_train = user_train.get(user_id, [])[-15:]
    for mid, rating in recent_train:
        info = movie_info.get(mid, {})
        history.append({
            'id': mid,
            'title': info.get('Title', ''),
            'genres': info.get('Genre', []),
            'user_rating': rating
        })
    return history

def get_candidate_details(user_id: int) -> List[Dict]:
    """Reused from original script."""
    candidates = []
    for mid in user_recommendations.get(user_id, []):
        info = movie_info.get(mid, {})
        candidates.append({
            'id': mid,
            'title': info.get('Title', ''),
            'genres': info.get('Genre', [])
        })
    return candidates

# ----------------------
# 7. Rerank Pipeline Using the Baseline Model
# ----------------------
def rerank_for_user_baseline(user_id: int) -> List[int]:
    """
    Executes the single-call, few-shot baseline model.
    """
    print(f"\n--- Reranking for User {user_id} using Few-Shot Baseline ---")

    # 1. Get and format history and candidates
    history_list = get_user_history(user_id)
    history_str = format_history_for_prompt(history_list)

    candidates_list = get_candidate_details(user_id)
    candidates_str = format_candidates_for_prompt(candidates_list)

    # 2. Run the single-call chain
    reranked_list_raw = baseline_chain.run(history=history_str, candidates=candidates_str)
    print(f"LLM Raw Output: {reranked_list_raw}")

    # 3. Parse the output
    try:
        final_ranking = json.loads(reranked_list_raw)
    except Exception:
        print("Warning: LLM output was not valid JSON. Attempting to parse manually.")
        final_ranking = [int(x) for x in reranked_list_raw.replace('[', '').replace(']', '').split(',') if x.strip().isdigit()]

    return final_ranking

# ----------------------
# 8. Evaluation Logic
# ----------------------
if __name__ == '__main__':
    overall_hits = head_hits = tail_hits = 0
    head_total = tail_total = 0
    overall_ndcg_sum = head_ndcg_sum = tail_ndcg_sum = 0.0

    successfully_processed = 0
    sample_size = 200  # Adjust as needed

    user_items = list(user_labels.items())
    if len(user_items) > sample_size:
        random.seed(55)
        sampled_user_items = random.sample(user_items, sample_size)
    else:
        sampled_user_items = user_items

    total_sampled = len(sampled_user_items)
    print(f"Sampling {total_sampled} users for this run...")

    for i, (uid, test_mid) in enumerate(sampled_user_items):
        try:
            final_ranking = rerank_for_user_baseline(uid)

            original_candidate_ids = {c['id'] for c in get_candidate_details(uid)}
            final_ranking_cleaned = [mid for mid in final_ranking if mid in original_candidate_ids]

            successfully_processed += 1

            popularity = movie_popularity.get(test_mid, 'L')
            if popularity == 'H':
                head_total += 1
            else:
                tail_total += 1

            ndcg_score = 0.0
            if test_mid in final_ranking_cleaned[:10]:
                overall_hits += 1
                if popularity == 'H':
                    head_hits += 1
                else:
                    tail_hits += 1
                rank_index = final_ranking_cleaned[:10].index(test_mid)
                ndcg_score = 1 / math.log2(rank_index + 2)
                print(
                    f"✅ User {uid} Hit@10 (Type: {popularity}): {test_mid} found at rank {rank_index + 1}. NDCG: {ndcg_score:.4f}")
            else:
                print(f"❌ User {uid} Miss@10 (Type: {popularity}): {test_mid} not found in top 10.")

            overall_ndcg_sum += ndcg_score
            if popularity == 'H':
                head_ndcg_sum += ndcg_score
            else:
                tail_ndcg_sum += ndcg_score

        except Exception as e:
            print(f"--- ‼️ Error processing user {uid}, skipping. Reason: {e} ---")
            continue

        if successfully_processed > 0 and successfully_processed % 20 == 0:
            current_overall_hr = overall_hits / successfully_processed
            current_head_hr = head_hits / head_total if head_total > 0 else 0
            current_tail_hr = tail_hits / tail_total if tail_total > 0 else 0
            current_overall_ndcg = overall_ndcg_sum / successfully_processed
            current_head_ndcg = head_ndcg_sum / head_total if head_total > 0 else 0
            current_tail_ndcg = tail_ndcg_sum / tail_total if tail_total > 0 else 0

            print(f"\n--- Progress: {i + 1}/{total_sampled} checked. "
                  f"{successfully_processed} processed.\n"
                  f" Interim HR@10 -> Overall: {current_overall_hr:.4f}, Head: {current_head_hr:.4f}, Tail: {current_tail_hr:.4f}\n"
                  f" Interim NDCG@10 -> Overall: {current_overall_ndcg:.4f}, Head: {current_head_ndcg:.4f}, Tail: {current_tail_ndcg:.4f} ---\n")

    # --- FINAL Report ---
    overall_hr = overall_hits / successfully_processed if successfully_processed > 0 else 0
    head_hr = head_hits / head_total if head_total > 0 else 0
    tail_hr = tail_hits / tail_total if tail_total > 0 else 0

    overall_ndcg = overall_ndcg_sum / successfully_processed if successfully_processed > 0 else 0
    head_ndcg = head_ndcg_sum / head_total if head_total > 0 else 0
    tail_ndcg = tail_ndcg_sum / tail_total if tail_total > 0 else 0

    failed_count = total_sampled - successfully_processed

    print("\n" + "="*50)
    print("FINAL EVALUATION REPORT (FEW-SHOT BASELINE)")
    print("-"*50)
    print(f"Total Users Sampled: {total_sampled}")
    print(f"Successfully Processed: {successfully_processed}")
    print(f"Failed to Process: {failed_count}")
    print("-"*50)
    print(f"Total Hits@10 (Overall): {overall_hits} (out of {successfully_processed} users)")
    print(f" - Head Item Hits: {head_hits} (out of {head_total} head items)")
    print(f" - Tail Item Hits: {tail_hits} (out of {tail_total} tail items)")
    print("."*50)
    print(f"FINAL HR@10 (Overall): {overall_hr:.4f}")
    print(f"FINAL Head HR@10: {head_hr:.4f}")
    print(f"FINAL Tail HR@10: {tail_hr:.4f}")
    print("."*50)
    print(f"FINAL NDCG@10 (Overall): {overall_ndcg:.4f}")
    print(f"FINAL Head NDCG@10: {head_ndcg:.4f}")
    print(f"FINAL Tail NDCG@10: {tail_ndcg:.4f}")
    print("="*50)
