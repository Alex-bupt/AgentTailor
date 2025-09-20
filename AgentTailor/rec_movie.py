import json
import math
import os
import random
import re
from collections import defaultdict
from typing import List, Dict, Tuple

import pandas as pd
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# ----------------------
# 1. API Configuration
# ----------------------
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"
os.environ["OPENAI_API_KEY"] = "your API Key"  # Please replace with your API Key

model_name = "deepseek-chat"

# ----------------------
# 2. File Paths
# ----------------------
RATINGS_FILE = "./ml-1m new/ratings_cleaned.txt"
USER_RECS_CSV = "./ml-1m new/user_recommendations.csv"
USER_LABELS_CSV = "./ml-1m new/user_labels.csv"
MOVIES_JSON = "./ml-1m new/movies_filled.json"
MOVIE_POPULARITY_CSV = "./ml-1m new/movie_popularity.csv"

# ----------------------
# 3. Load Data from Files
# ----------------------
# Load ratings data
user_ratings_all: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
with open(RATINGS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("::")
        if len(parts) < 4: continue
        uid, mid, rating, _ = int(parts[0]), int(parts[1]), float(parts[2]), parts[3]
        user_ratings_all[uid].append((mid, rating))

user_train: Dict[int, List[Tuple[int, float]]] = {uid: hist[:-1] for uid, hist in user_ratings_all.items() if hist}
user_test: Dict[int, Tuple[int, float]] = {uid: hist[-1] for uid, hist in user_ratings_all.items() if hist}

# Load pre-calculated Top-20 recommendations
user_recs_df = pd.read_csv(USER_RECS_CSV)
user_recommendations: Dict[int, List[int]] = {
    int(row['UserID']): [int(row[f'Rec{i + 1}']) for i in range(20)]
    for _, row in user_recs_df.iterrows()
}

# Load test labels
labels_df = pd.read_csv(USER_LABELS_CSV)
user_labels: Dict[int, int] = {int(row['UserID']): int(row['TestMovieID']) for _, row in labels_df.iterrows()}

# Load movie metadata
with open(MOVIES_JSON, 'r', encoding='utf-8') as f:
    movies_data = json.load(f)
movie_info: Dict[int, Dict] = {int(item['MovieID']): item for item in movies_data}

# Load movie popularity data
popularity_df = pd.read_csv(MOVIE_POPULARITY_CSV)
movie_popularity: Dict[int, str] = {int(row['MovieID']): row['Popularity'] for _, row in popularity_df.iterrows()}

# ----------------------
# 4. LLM Client Setup
# ----------------------
llm = ChatOpenAI(model_name=model_name, temperature=1.0)

# -----------------------------------
# 5. Define Agent Prompt Templates
# -----------------------------------

# === MODIFIED AGENT 1 (Profile Generation Only) ===
prompt_1 = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """You are a highly-disciplined user analyst model. Your task is to analyze a user's viewing history and output a rich, structured user profile.

### INSTRUCTIONS:
Your goal is to create a detailed user profile object with the following components:

1.  **`positive_preferences`**: Identify what the user likes.
    -   **`genres`, `actors`, `directors`**: Extract the user's most frequent and highly-rated genres, actors, and directors into simple lists of strings.
    -   **`themes_keywords`**: Extract a list of recurring themes, tones, or keywords from the descriptions of movies the user liked (e.g., "fast-paced", "dystopian", "mind-bending", "1990s").

2.  **`negative_preferences`**: Identify what the user dislikes by finding genres, actors, or directors that are consistently rated very low (e.g., rating <= 2) and list them.

3.  **`summary`**: Write a concise, 3-5 sentence summary that synthesizes all the findings above.

### OUTPUT FORMATTING RULES:
- You MUST output a single JSON object enclosed in triple backticks ` ```json ... ``` `.
- The JSON object must contain a single top-level key: `user_profile`.

### EXAMPLE OF YOUR REQUIRED OUTPUT:
```json
{{
  "user_profile": {{
    "summary": "This user strongly prefers 1990s sci-fi thrillers directed by James Cameron, showing a secondary interest in fast-paced action movies. They actively dislike slow-paced dramas and horror films. The recurring themes in their viewing history are dystopian futures and high-stakes conflicts.",
    "positive_preferences": {{
      "genres": ["Sci-Fi", "Thriller", "Action"],
      "actors": ["Harrison Ford"],
      "directors": ["James Cameron"],
      "themes_keywords": ["dystopian", "fast-paced", "alien encounters", "1990s"]
    }},
    "negative_preferences": {{
      "genres": ["Drama", "Horror"]
    }}
  }}
}}
```"""
    ),
    HumanMessagePromptTemplate.from_template(
        "Analyze the following user viewing history:\n{history}\n\n"
        "Reference Movie Popularity Data (for context):\n{popularity_info}\n\n"
    ),
])
agent1_chain_profile_only = LLMChain(llm=llm, prompt=prompt_1, output_key="agent1_output")

# Agent 2: Candidate Movie Parsing (Relevance Evaluation) - Unchanged
prompt_2 = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """You are a detailed Movie Compatibility Analyzer. Your task is to analyze each candidate movie and break down its compatibility with the detailed, structured user profile you have been given.

### INSTRUCTIONS:
- For each candidate movie, you will evaluate it against the user's `positive_preferences`, `negative_preferences`, and `themes_keywords`.
- You must score the compatibility for each of the following four dimensions on a scale from 0.0 (no match) to 1.0 (perfect match).
- For each score, you must provide a brief, one-sentence `reasoning` that explicitly references the user's profile.
- **Crucially, if a movie matches a `negative_preference`, its corresponding score should be very low (e.g., 0.0 or 0.1).**

### DIMENSIONS TO EVALUATE:
1.  **`genre_match`**: Compare the movie's genres to the `positive_preferences.genres` and `negative_preferences.genres`.
2.  **`actor_match`**: Compare the movie's actors to the `positive_preferences.actors`.
3.  **`director_match`**: Compare the movie's director to the `positive_preferences.directors`.
4.  **`vibe_match`**: Compare the movie's description and themes to the `positive_preferences.themes_keywords` and the overall `summary`.

### OUTPUT FORMAT:
- You MUST output a single JSON object containing a key "analyzed_candidates".
- This key will contain a list of objects, where each object represents a movie you have analyzed.
- Do not include any other text or explanations.

### EXAMPLE OF YOUR REQUIRED OUTPUT:
```json
{{
  "analyzed_candidates": [
    {{
      "id": 858,
      "title": "Godfather, The",
      "compatibility_breakdown": {{
        "genre_match": {{
          "score": 0.2,
          "reasoning": "The 'Drama' genre matches a known negative preference for the user."
        }},
        "actor_match": {{
          "score": 0.0,
          "reasoning": "No preferred actors are present."
        }},
        "director_match": {{
          "score": 0.0,
          "reasoning": "The director is not on the user's preferred list."
        }},
        "vibe_match": {{
          "score": 0.3,
          "reasoning": "The crime drama theme does not align with the user's preferred 'dystopian' and 'fast-paced' keywords."
        }}
      }}
    }},
    {{
      "id": 2571,
      "title": "Matrix, The",
      "compatibility_breakdown": {{
        "genre_match": {{
          "score": 1.0,
          "reasoning": "Matches both 'Sci-Fi' and 'Action', which are strong positive preferences."
        }},
        "actor_match": {{
          "score": 0.0,
          "reasoning": "No preferred actors are present."
        }},
        "director_match": {{
          "score": 0.0,
          "reasoning": "The directors are not on the user's preferred list."
        }},
        "vibe_match": {{
          "score": 0.9,
          "reasoning": "The movie's themes strongly align with the user's preferred keywords 'dystopian' and 'fast-paced'."
        }}
      }}
    }}
  ]
}}
```"""
    ),
    HumanMessagePromptTemplate.from_template(
        "User Profile:\n{user_profile}\n\nCandidate Movies:\n{candidates}\n"
    ),
])
agent2_chain = LLMChain(llm=llm, prompt=prompt_2, output_key="parsed_candidates")

# Agent 3 (New): Deliberative Ranking Agent
prompt_3 = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """You are a Ranking Review Agent. Your mission is to review a pre-existing analysis of candidate movies and create a final ranking based *only* on that analysis.

### Core Mission
Based *exclusively* on the provided 'Parsed Candidate Analysis', create the most logical ranking. You do not have access to the full user profile or full movie details.

### Ranking Process (You MUST follow these steps):
1.  **Assess Analysis Quality**: For each candidate, evaluate the `analysis` text provided. A high `score` is more trustworthy if the `analysis` provides specific and compelling reasons for the match. A high `score` with vague reasoning is less reliable.

2.  **Identify Key Signals in Analysis**:
    - **Positive Signals**: Look for phrases in the `analysis` text that suggest an exceptionally strong match (e.g., "perfectly aligns with...", "matches multiple key aspects..."). These movies should be promoted.
    - **Negative Signals**: Look for phrases that indicate potential mismatches or drawbacks, even if the final `score` is high (e.g., "however, the theme is a slight mismatch...", "lacks the user's favorite actor..."). These movies should be demoted.

3.  **Construct Final Ranking**:
    - Synthesize the numeric `score` and your assessment of the `analysis` text to build the final ordered list of movie IDs. Your goal is to produce a ranking that best reflects the "spirit" of the provided analysis.

### Output Format
- **You MUST output a single JSON array containing the final, ordered list of movie IDs.**
- **Do NOT include any other text, explanations, or commentary before or after the JSON array.**
"""
    ),
    HumanMessagePromptTemplate.from_template(
        "Perform a deliberative ranking based on the following analysis:\n\n"
        "**Parsed Candidate Analysis:**\n{parsed_candidates}\n\n"
        "**Your Final Deliberated Ranking:**"
    ),
])

agent3_deliberative_chain = LLMChain(llm=llm, prompt=prompt_3, output_key="final_ranking")

# === MODIFIED AGENT 4 ===
# Agent 4 - Guided Causal & Personalized Debiasing Agent
prompt_4_guided = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """You are a Causal & Personalized Debiasing Agent. Your task is to re-rank a movie list to intelligently blend user relevance with novelty, guided by the `lambda_u` value.

**Core Re-ranking Principle:**
Your goal is to adjust the initial relevance-based ranking to introduce novelty (promoting niche 'L' items) or reinforce popular choices (promoting 'H' items) based on `lambda_u`.

**Re-ranking Rules (Apply in order):**
1.  **Anchor the Top Relevance**: The top 3-5 movies from the input ranking are considered **highly relevant**. Do NOT demote these movies unless the justification is exceptionally strong (e.g., it's an 'H' movie and `lambda_u` is extremely high > 0.9). These are the user's likely favorites.

2.  **Apply `lambda_u` Adjustments**:
    * **High `lambda_u` (≥ 0.7 - Seeks Novelty)**: Actively look for 'L' (long-tail) movies in the middle or lower ranks that are a strong match to the user's `key_preferences`. Promote 1-3 of these movies into the top 10, potentially swapping them with lower-ranked 'H' movies that are only a moderate match.
    * **Medium `lambda_u` (0.3–0.7 - Balanced)**: Cautiously promote one or two highly relevant 'L' movies into the top 10. The final list should be a healthy mix of 'H' and 'L' items, but the highest ranks should still be dominated by the most relevant movies regardless of popularity.
    * **Low `lambda_u` (< 0.3 - Prefers Popular)**: Make minimal changes. The ranking should closely follow the original relevance order. Only consider promoting an 'L' movie if it is an *absolutely perfect* match for the user's profile and a higher-ranked 'H' item is a weak match.

3.  **Preserve Overall Coherence**: After making swaps, ensure the final list is logical. A movie's final position should be a balance of its original relevance and the novelty adjustment. Do not make drastic changes that push a top-5 movie to the bottom of the list.

**Output Format:**
Return the final re-ranked movie IDs as a JSON array (e.g., `[ID1, ID2, ID3, ...]`) with **no extra text**."""
    ),
    HumanMessagePromptTemplate.from_template(
        "Guiding Debiasing Strength (lambda_u): {lambda_u}\n\n"
        "User Profile:\n{user_profile}\n\n"
        "Ranking to be Refined (from Agent 3):\n{final_ranking}\n\n"
        "Movie Popularity Data:\n{popularity_info}\n\n"
        "Full Candidate Movie Details:\n{candidate_details}\n\n"
        "Your Final Debiased Ranking:"
    ),
])
agent4_guided_chain = LLMChain(llm=llm, prompt=prompt_4_guided, output_key="debiased_ranking")


# -----------------------------------
# 6. Helper Functions to Build Inputs
# -----------------------------------
def get_user_history(user_id: int) -> List[Dict]:
    history = []
    # Use last 50 ratings for history
    for mid, rating in user_train.get(user_id, [])[-50:]:
        info = movie_info.get(mid, {})
        history.append({
            'id': mid, 'title': info.get('Title', ''), 'genres': info.get('Genre', []),
            'actors': info.get('Actors', []), 'director': info.get('Director', ''),
            'description': info.get('Description', ''), 'user_rating(5 point scale)': rating
        })
    return history


def get_candidate_details(user_id: int) -> List[Dict]:
    candidates = []
    for mid in user_recommendations.get(user_id, []):
        info = movie_info.get(mid, {})
        candidates.append({
            'id': mid, 'title': info.get('Title', ''), 'genres': info.get('Genre', []),
            'actors': info.get('Actors', []), 'director': info.get('Director', ''),
            'description': info.get('Description', '')
        })
    return candidates


def get_popularity_info(movie_ids: List[int]) -> str:
    info_lines = []
    for mid in movie_ids:
        title = movie_info.get(mid, {}).get('Title', 'Unknown Title')
        pop_type = movie_popularity.get(mid, 'L')
        info_lines.append(f"- ID: {mid}, Title: {title}, Popularity: {pop_type}")
    return "\n".join(info_lines)


# -----------------------------------
# 7. Rerank Pipeline Using Agents (Updated)
# -----------------------------------
# <<< CHANGE 1: Update the return type hint >>>
def rerank_for_user(user_id: int) -> Tuple[List[int], List[int]]:
    """
    Executes the full agent pipeline, returning both Agent 3 and Agent 4's rankings.
    """
    history_info = get_user_history(user_id)
    history_str = json.dumps(history_info, indent=2)
    print(f"\n--- Reranking for User {user_id} with {len(history_info)} history items ---")

    # === Manually calculate lambda_u from user's viewing history ===
    head_count = 0
    tail_count = 0
    if history_info:
        for movie in history_info:
            mid = movie.get('id')
            if mid and mid in movie_popularity:
                pop_type = movie_popularity.get(mid)
                if pop_type == 'H':
                    head_count += 1
                else:
                    tail_count += 1

        total_watched = head_count + tail_count
        lambda_u = (tail_count / total_watched) if total_watched > 0 else 0.5
    else:
        lambda_u = 0.5

    print(f"Manually calculated lambda_u: {lambda_u:.2f} (Head: {head_count}, Tail: {tail_count})")

    candidates = get_candidate_details(user_id)
    candidates_str = json.dumps(candidates, indent=2)

    # === STEP 1: Agent 1 generates profile ONLY ===
    user_profile = {"summary": "A general movie watcher.", "key_preferences": {}}
    history_popularity_info = get_popularity_info([m['id'] for m in history_info])

    try:
        agent1_output_str = agent1_chain_profile_only.run(
            history=history_str,
            popularity_info=history_popularity_info,
        )
        match = re.search(r"```json\s*(\{.*?\})\s*```", agent1_output_str, re.DOTALL)
        json_str_to_parse = match.group(1) if match else agent1_output_str
        agent1_data = json.loads(json_str_to_parse)
        user_profile = agent1_data.get('user_profile', user_profile)

    except Exception as e:
        print(f"Error parsing Agent 1 output: {e}. Falling back to default profile.")
        print(f"RAW agent1 output: {agent1_output_str}")

    # === STEP 2: Agent 2 parses candidates based on profile ===
    parsed_candidates = agent2_chain.run(
        user_profile=json.dumps(user_profile, indent=2),
        candidates=candidates_str
    )

    # === STEP 3: Agent 3 creates a deliberated ranking ===
    final_ranking_raw_agent3 = agent3_deliberative_chain.run(
        user_profile=json.dumps(user_profile, indent=2),
        parsed_candidates=parsed_candidates,
        candidate_details=candidates_str
    )
    try:
        deliberated_ranking = json.loads(final_ranking_raw_agent3)
    except Exception:
        deliberated_ranking = [int(x) for x in
                               final_ranking_raw_agent3.strip().replace('[', '').replace(']', '').split(',') if
                               x.strip().isdigit()]

    # === STEP 4: Agent 4 performs debiasing guided by the manually calculated lambda_u ===
    candidate_popularity_info = get_popularity_info([c['id'] for c in candidates])
    debiased_ranking_raw = agent4_guided_chain.run(
        lambda_u=lambda_u,
        user_profile=json.dumps(user_profile, indent=2),
        final_ranking=json.dumps(deliberated_ranking),
        popularity_info=candidate_popularity_info,
        candidate_details=candidates_str
    )
    print(f"Agent 3 (Relevance) -> List: {deliberated_ranking}")
    print(f"Agent 4 (Debiased) -> Final List: {debiased_ranking_raw}")

    # Parse the final output for evaluation
    try:
        final_ranking = json.loads(debiased_ranking_raw)
    except Exception:
        final_ranking = [int(x) for x in debiased_ranking_raw.strip().replace('[', '').replace(']', '').split(',') if
                         x.strip().isdigit()]

    # <<< CHANGE 2: Return both rankings >>>
    return deliberated_ranking, final_ranking


# -----------------------------------
# 8. Evaluation Logic
# -----------------------------------
# <<< CHANGE 3: Update the function docstring and logic >>>
def eval_user(uid: int, test_mid: int):
    """
    Returns a tuple (hit, pop, ndcg, rerank_fail, debiasing_fail) or None on failure.
    rerank_fail == 1 if test_mid is in the top-20 but not in the top-10.
    debiasing_fail == 1 if test_mid is in Agent3's top-10 but not in Agent4's top-10.
    """
    try:
        initial_candidates = user_recommendations.get(uid, [])
        if not initial_candidates or test_mid not in initial_candidates:
            print(f"Skipping user {uid}: test movie {test_mid} not in initial top-20 candidates.")
            return None

        # Get both rankings
        agent3_ranking, agent4_ranking = rerank_for_user(uid)
        if not agent3_ranking or not agent4_ranking:
            return None

        # --- Metrics based on final Agent 4 ranking ---
        final_ranking_20 = agent4_ranking
        final_ranking_10 = final_ranking_20[:10]
        pop = movie_popularity.get(test_mid, 'L')

        hit = int(test_mid in final_ranking_10)
        ndcg = 0.0
        if hit:
            idx = final_ranking_10.index(test_mid)
            ndcg = 1.0 / math.log2(idx + 2)

        rerank_fail = int((test_mid in final_ranking_20) and (not hit))

        # --- NEW METRIC: Debiasing Failure ---
        agent3_hit = int(test_mid in agent3_ranking[:10])
        debiasing_fail = 1 if agent3_hit and not hit else 0

        return hit, pop, ndcg, rerank_fail, debiasing_fail

    except Exception as e:
        print(f"An unhandled exception occurred for user {uid}: {e}")
        return None

# -----------------------------------
# 8. Evaluation Logic
# -----------------------------------
if __name__ == '__main__':
    overall_hits, head_hits, tail_hits = 0, 0, 0
    head_total, tail_total = 0, 0
    overall_ndcg_sum, head_ndcg_sum, tail_ndcg_sum = 0.0, 0.0, 0.0
    rerank_failures = 0 # <<<< NEW: Counter for rerank failures

    successfully_processed = 0
    sample_size = 200 # Adjust as needed

    user_items = list(user_labels.items())
    if len(user_items) > sample_size:
        random.seed(66) # Use a fixed seed for reproducibility
        sampled_user_items = random.sample(user_items, sample_size)
    else:
        sampled_user_items = user_items

    total_sampled = len(sampled_user_items)
    print(f"Sampling {total_sampled} users for this run...")

    for i, (uid, test_mid) in enumerate(sampled_user_items):
        try:
            # The initial Top-20 list from the base model
            initial_candidates = user_recommendations.get(uid, [])
            # The final list after re-ranking by the agents
            final_ranking = rerank_for_user(uid)
            successfully_processed += 1

            popularity = movie_popularity.get(test_mid, 'L')
            if popularity == 'H':
                head_total += 1
            else:
                tail_total += 1

            ndcg_score = 0.0
            # Check for a hit in the Top-10
            if test_mid in final_ranking[:10]:
                overall_hits += 1
                if popularity == 'H': head_hits += 1
                else: tail_hits += 1

                rank_index = final_ranking[:10].index(test_mid)
                ndcg_score = 1 / math.log2(rank_index + 2)
                print(f"✅ User {uid} Hit@10 (Type: {popularity}): {test_mid} found at rank {rank_index + 1}. NDCG: {ndcg_score:.4f}")
            else:
                # <<<< NEW: Logic to count rerank failures >>>>
                # This is an HR@10 miss. Since the test movie is guaranteed to be in the initial 20 candidates,
                # if it's not in the final top 10, it's a "rerank failure".
                rerank_failures += 1
                print(f"❌ User {uid} Miss@10 (Type: {popularity}): {test_mid} not in top 10. Rerank Failure Counted.")

            overall_ndcg_sum += ndcg_score
            if popularity == 'H': head_ndcg_sum += ndcg_score
            else: tail_ndcg_sum += ndcg_score

        except Exception as e:
            print(f"--- ‼️ Error processing user {uid}, skipping. Reason: {e} ---")
            continue

        # Progress reporting
        if successfully_processed > 0 and successfully_processed % 20 == 0:
            current_overall_hr = overall_hits / successfully_processed
            current_head_hr = head_hits / head_total if head_total > 0 else 0
            current_tail_hr = tail_hits / tail_total if tail_total > 0 else 0
            current_overall_ndcg = overall_ndcg_sum / successfully_processed
            current_head_ndcg = head_ndcg_sum / head_total if head_total > 0 else 0
            current_tail_ndcg = tail_ndcg_sum / tail_total if tail_total > 0 else 0

            print(f"\n--- Progress: {i + 1}/{total_sampled} checked. "
                  f"{successfully_processed} processed.\n"
                  f"    Interim HR@10   -> Overall: {current_overall_hr:.4f}, Head: {current_head_hr:.4f}, Tail: {current_tail_hr:.4f}\n"
                  f"    Interim NDCG@10 -> Overall: {current_overall_ndcg:.4f}, Head: {current_head_ndcg:.4f}, Tail: {current_tail_ndcg:.4f} ---\n")

    # Final Report
    if successfully_processed > 0:
        overall_hr = overall_hits / successfully_processed
        head_hr = head_hits / head_total if head_total > 0 else 0
        tail_hr = tail_hits / tail_total if tail_total > 0 else 0
        overall_ndcg = overall_ndcg_sum / successfully_processed
        head_ndcg = head_ndcg_sum / head_total if head_total > 0 else 0
        tail_ndcg = tail_ndcg_sum / tail_total if tail_total > 0 else 0
    else:
        overall_hr = head_hr = tail_hr = overall_ndcg = head_ndcg = tail_ndcg = 0.0

    print("\n==========================================================")
    print("FINAL EVALUATION REPORT")
    print("----------------------------------------------------------")
    print(f"Total Users Sampled:    {total_sampled}")
    print(f"Successfully Processed: {successfully_processed}")
    print(f"Failed to Process:      {total_sampled - successfully_processed}")
    print("----------------------------------------------------------")
    print(f"Total Hits@10 (Overall): {overall_hits} (out of {successfully_processed} users)")
    print(f"  - Head Item Hits:      {head_hits} (out of {head_total} head items)")
    print(f"  - Tail Item Hits:      {tail_hits} (out of {tail_total} tail items)")
    print(f"Re-rank Failures:        {rerank_failures} (HR@10 Miss but was in Top-20)") # <<<< NEW: Final report line
    print("..........................................................")
    print(f"FINAL HR@10 (Overall):   {overall_hr:.4f}")
    print(f"FINAL Head HR@10:        {head_hr:.4f}")
    print(f"FINAL Tail HR@10:        {tail_hr:.4f}")
    print("..........................................................")
    print(f"FINAL NDCG@10 (Overall): {overall_ndcg:.4f}")
    print(f"FINAL Head NDCG@10:      {head_ndcg:.4f}")
    print(f"FINAL Tail NDCG@10:      {tail_ndcg:.4f}")
    print("==========================================================")



