import json
import math
import os
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple

import pandas as pd
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from tqdm import tqdm

# ----------------------
# 1. API Configuration
# ----------------------
# Ensure that you have set environment variables or provide them here
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"
os.environ["OPENAI_API_KEY"] = "your API Key"  # ################### Replace with your API Key ###################

model_name = "deepseek-chat"

# ----------------------
# 2. File Paths & Constants
# ----------------------

RATINGS_FILE = "./bookcrossing new/ratings_processed.txt"
USER_RECS_CSV = "./bookcrossing new/user_top150_recommendations.csv"
BOOKS_JSON = "./bookcrossing new/filtered_books.json"
BOOK_POPULARITY_CSV = "./bookcrossing new/popularity.csv"

CANDIDATE_COUNT = 150
EVALUATION_K = 100
BATCH_SIZE = 15

# ----------------------
# 3. Output Schema (for Agent2, not strictly enforced)
# ----------------------
schemas = [
    ResponseSchema(
        name="analyzed_candidates",
        description=(
            "A list of analysis results, each with:\n"
            "- isbn (string)\n"
            "- title (string)\n"
            "- compatibility_breakdown (object) containing:\n"
            "    - genre_match: { score: float, reasoning: string }\n"
            "    - author_match: { score: float, reasoning: string }\n"
            "    - theme_match: { score: float, reasoning: string }\n"
            "    - context_match: { score: float, reasoning: string }"
        )
    ),
]
parser = StructuredOutputParser.from_response_schemas(schemas)

# ----------------------
# 4. Load Data from Files
# ----------------------
# Load rating data
user_ratings_all: Dict[int, List[Tuple[str, float]]] = defaultdict(list)
with open(RATINGS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(";")
        if len(parts) < 3:
            continue
        uid, isbn, rating_str = parts
        try:
            user_ratings_all[int(uid)].append((isbn, float(rating_str)))
        except ValueError:
            continue

# Split into training and test sets
user_train: Dict[int, List[Tuple[str, float]]] = {
    uid: hist[:-1] for uid, hist in user_ratings_all.items() if len(hist) > 1
}
user_test: Dict[int, Tuple[str, float]] = {
    uid: hist[-1] for uid, hist in user_ratings_all.items() if len(hist) > 1
}

# Load precomputed Top-150 recommendations
user_recs_df = pd.read_csv(USER_RECS_CSV)
user_recommendations: Dict[int, List[str]] = {
    int(row["UserID"]): [str(row[f"Rec{i + 1}"]) for i in range(CANDIDATE_COUNT)]
    for _, row in user_recs_df.iterrows()
}

# Load book metadata (ISBN as key)
with open(BOOKS_JSON, "r", encoding="utf-8") as f:
    books_data = json.load(f)
book_info: Dict[str, Dict] = {item["ISBN"]: item for item in books_data}

# Load book popularity data
popularity_df = pd.read_csv(BOOK_POPULARITY_CSV)
book_popularity: Dict[str, str] = pd.Series(
    popularity_df.Popularity_Class.values,
    index=popularity_df.ISBN.astype(str),
).to_dict()

# ----------------------
# 5. LLM Client Setup
# ----------------------
llm = ChatOpenAI(model_name=model_name, temperature=1.0)

# -----------------------------------
# 6. Define Agent Prompt Templates
# -----------------------------------

# Agent 1: User Profile Extraction for BookCrossing
prompt_1 = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            """You are a highly-disciplined user analyst model. Your task is to analyze a user's reading history and output a rich, structured user profile.

### INSTRUCTIONS:
1. **summary**: Write a 3–5 sentence overview synthesizing the user's reading taste, including key affinities and aversions.
2. **positive_preferences**: Extract what the user consistently rates highly. Break this into:
   - `genres` (e.g., "Science Fiction", "Historical Fiction"),
   - `authors`,
   - `themes` (recurring narrative or conceptual themes, e.g., "coming-of-age", "dystopia"),
   - `audience` (e.g., "Young Adult", "Academic", inferred preferred target readership),
   - `settings` (e.g., "near-future", "alternate history", "small-town").
3. **negative_preferences**: Identify genres, authors, or themes the user rates very low (rating threshold: <=4/10) and list them under the appropriate keys.
4. **reading_habits_summary**: Note any notable patterns such as diversity of genres, tendency toward series vs. standalones, temporal recency bias, pacing preferences, etc.

### OUTPUT FORMATTING RULES:
- You MUST output a single JSON object enclosed in triple backticks ```json ... ```.
- The top-level key must be `user_profile`.
- All subfields should be present; if some category has no data, use empty lists or an empty string as appropriate.

### EXAMPLE OF REQUIRED OUTPUT:
```json
{
  "user_profile": {
    "summary": "The user prefers speculative fiction with strong world-building and coming-of-age arcs, showing a secondary interest in alternate-history political thrillers. They tend to avoid slow-paced romances and overly sentimental dramas. Their reading habits show a balance between series and standalone novels, with a preference for near-future and slightly dystopian settings.",
    "positive_preferences": {
      "genres": ["Science Fiction", "Speculative Fiction"],
      "authors": ["N.K. Jemisin", "Brandon Sanderson"],
      "themes": ["world-building", "coming-of-age", "social justice"],
      "audience": ["Young Adult", "Adult"],
      "settings": ["Near-future", "Alternate History"]
    },
    "negative_preferences": {
      "genres": ["Romance"],
      "authors": ["Author X"],
      "themes": ["slow-paced", "overly sentimental"]
    },
    "reading_habits_summary": "Reads both series and standalones, favors diverse settings, and shows a mild recent bias toward speculative works."
  }
}
```"""
        ),
        HumanMessagePromptTemplate.from_template(
            "Analyze the following user reading history:\n{history}\n\nReference Popularity Data:\n{popularity_info}\n"
        ),
    ]
)
agent1_chain_profile_only = LLMChain(llm=llm, prompt=prompt_1, output_key="agent1_output")

# Agent 2: Candidate Book Compatibility Analyzer for BookCrossing
prompt_2 = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            """You are a detailed Book Compatibility Analyzer. Your task is to analyze each candidate book and break down its compatibility with the detailed, structured user profile you have been given.

### INSTRUCTIONS:
- For each candidate book, evaluate the following four dimensions and assign a score from 0.0 (no match) to 1.0 (perfect match).
- Provide a brief, one-sentence `reasoning` for each score that explicitly references the relevant parts of the user's profile.
- **If a book matches any `negative_preferences` (e.g., genre, author, or theme), the corresponding score must be very low (0.0 or 0.1).**

### DIMENSIONS TO EVALUATE:
1. **`genre_match`**: How well the book's genres align with `positive_preferences.genres` and avoid `negative_preferences.genres`.
2. **`author_match`**: Whether the author is among preferred authors or in the negative list.
3. **`theme_match`**: Alignment between the book's core themes and the user's `positive_preferences.themes` (or conflict with negative themes).
4. **`context_match`**: Fit in terms of setting/audience/overall vibe (e.g., whether the book's setting and intended readership reflect `positive_preferences.settings` and `audience`).

### OUTPUT FORMAT:
- You MUST output a single JSON object containing the key `"analyzed_candidates"`.
- Its value is a list of objects; each object must include:
  - `id`: candidate book identifier,
  - `title`: book title,
  - `compatibility_breakdown`: an object with the four dimensions, each having `score` and `reasoning`.

### EXAMPLE OF REQUIRED OUTPUT:
```json
{
  "analyzed_candidates": [
    {
      "id": 101,
      "title": "Futurebound",
      "compatibility_breakdown": {
        "genre_match": {
          "score": 0.9,
          "reasoning": "The book is Science Fiction, matching the user's strong positive preference for speculative genres."
        },
        "author_match": {
          "score": 0.0,
          "reasoning": "The author is listed in the user's negative preferences."
        },
        "theme_match": {
          "score": 0.8,
          "reasoning": "Themes of societal upheaval align with the user's interest in social justice and world-building."
        },
        "context_match": {
          "score": 0.7,
          "reasoning": "Near-future setting and adult audience fit the user's preferred settings and readership."
        }
      }
    }
  ]
}
```"""
        ),
        HumanMessagePromptTemplate.from_template(
            "User Profile:\n{user_profile}\n\nCandidate Books:\n{candidates}\n"
        ),
    ]
)
agent2_chain = LLMChain(llm=llm, prompt=prompt_2, output_key="parsed_candidates")

# === BookCrossing Agent 3: Deliberative Scoring Agent ===
prompt_3 = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            """You are a Ranking Review Agent for BookCrossing. Your task is to review the parsed candidate analyses (which include per-book compatibility_breakdown with scores and reasoning) and produce a refined, deliberated numeric score for each candidate. You do NOT have access to external book metadata or the full user profile—only the provided analysis text. Your output will be used downstream by an algorithm to sort candidates.

### CORE LOGIC:
1. **Assess Analysis Quality**: Start from the average of the four dimension scores (`genre_match`, `author_match`, `theme_match`, `context_match`). Then adjust that base score up or down based on the quality and signals in the reasoning.
   - **Positive signals** (should boost): concrete phrases such as "perfectly aligns with", "strong match on multiple key preferences", "clear synergy", "exceptionally relevant", multiple dimensions consistently high with specific evidence.
   - **Negative signals** (should penalize): hedge/contradictory language like "however", "slight mismatch", "despite", vague reasoning ("somewhat fits", "could work"), or indications that the candidate touches a negative preference.
2. **Adjustment Bound**: Adjust the base average by at most ±0.15 to form the final `deliberated_score`, clipping to stay within [0.0, 1.0].
3. **Explain Concisely**: For each candidate, include a brief summary of why its score was adjusted (e.g., "Boosted due to strong multi-dimension alignment with specific evidence" or "Slightly penalized because reasoning admits a thematic mismatch despite high genre score").

### OUTPUT FORMAT:
- Output a single JSON object with key `"refined_candidates"`.
- Its value is a list of objects; each object must contain:
  - `isbn`: the book's ISBN (string),
  - `title`: the book title (string),
  - `deliberated_score`: final float in [0.0,1.0],
  - `base_score`: the average of the four original dimension scores,
  - `adjustment_reasoning`: one-sentence explanation of the adjustment (positive or negative).

- Do **NOT** include any extra text outside the JSON.

### EXAMPLE OUTPUT:
```json
{
  "refined_candidates": [
    {
      "isbn": "9780261103573",
      "title": "Example Book",
      "deliberated_score": 0.88,
      "base_score": 0.75,
      "adjustment_reasoning": "Boosted because multiple dimensions have specific strong alignment and the reasoning uses phrases like 'perfectly matches' and 'key preferences'."
    },
    {
      "isbn": "9780143127741",
      "title": "Another Book",
      "deliberated_score": 0.42,
      "base_score": 0.55,
      "adjustment_reasoning": "Penalized due to vague reasoning and a noted thematic mismatch despite decent genre match."
    }
  ]
}
```"""
        ),
        HumanMessagePromptTemplate.from_template(
            "Perform a deliberative scoring based on the following parsed candidate analyses:\n\n"
            "**Parsed Candidate Analysis:**\n{parsed_candidates}\n\n"
            "**Your Refined Scores:**"
        ),
    ]
)
agent3_deliberative_chain = LLMChain(llm=llm, prompt=prompt_3, output_key="final_ranking")

# === BookCrossing Agent 4.1: Swap Strategy Agent (with ml1m-style ranking principles) ===
prompt_4_1_strategist = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            """You are a Swap Strategy Agent. Your task is to propose swaps that improve the current ranking by blending relevance with novelty, guided by lambda_u, using the swap-based framework.

### Core Swap Strategy Principles (adapted from the ml1m re-ranking rules):
1. **Anchor Top Relevance:** The top 3-5 items in the summarized ranked list are highly relevant; avoid proposing swaps that demote them unless there is exceptionally strong justification (e.g., they are overly popular 'H' items and lambda_u is very high and a niche 'L' item is a far better personalized match).
2. **Apply lambda_u Adjustments:**
   * **High lambda_u (≥ 0.7 – Seek Novelty):** Identify niche / long-tail ('L') books in middle or lower ranks that match the user's key preferences strongly and propose swapping them up into the top region by replacing higher-ranked popular ('H') books that are only moderate matches.
   * **Medium lambda_u (0.3–0.7 – Balanced):** Propose one or two cautious swaps that introduce highly relevant 'L' books into higher positions while preserving a mix; do not overdo novelty—keep the highest ranks dominated by relevance.
   * **Low lambda_u (< 0.3 – Prefers Popular):** Propose minimal or no swaps; only suggest swapping in an 'L' book if it is an absolutely perfect profile match and it replaces a weakly matching higher-ranked 'H' book.
3. **Preserve Coherence:** Swap proposals should be pairwise (promote one, demote one) and should not create jarring disruptions. Avoid chains of cascading swaps; each proposal should be justifiable independently.

### OUTPUT:
- A single JSON object with key `"swap_proposals"`.
- Value is a list of pairs `[["ISBN_to_promote", "ISBN_to_demote"], ...]`.
- Each pair means: promote the first ISBN and demote the second to improve the ranking under the above principles.
- Do NOT include any explanatory text or extra fields.

### EXAMPLE:
```json
{ "swap_proposals": [ ["9780261103573", "9780439023481"], ["9780553386790", "9780143127741"] ] }
```"""
        ),
        HumanMessagePromptTemplate.from_template(
            "User Profile:\n{user_profile}\n\nNovelty Preference (lambda_u): {lambda_u}\n\nSummarized Ranked List:\n{summarized_list}\n\nYour swap proposals:"
        ),
    ]
)
agent4_1_strategist_chain = LLMChain(llm=llm, prompt=prompt_4_1_strategist, output_key="swap_proposals_raw")

# === BookCrossing Agent 4.2: Swap Execution Agent ===
prompt_4_2_executor = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            """You are a Swap Execution Agent. Apply the proposed swaps to produce a final coherent re-ranked list of all candidate book ISBNs.

### INSTRUCTIONS:
1. **Apply swap proposals in order**: For each pair ["ISBN_promote", "ISBN_demote"], swap their positions in the initial ranking, resolving conflicts by honoring earlier proposals first if an ISBN appears in multiple swaps.
2. **Respect Anchors**: Do not drastically demote the original top 3-5 highly relevant items unless they were explicitly included in a swap proposal with strong implied justification; otherwise their relative ordering should largely remain.
3. **Maintain Coherence**: After executing swaps, ensure the full list remains a logical permutation of the initial ranking—only the suggested exchanges (and any necessary resolution of overlapping proposals) should alter positions. Do not introduce new arbitrary reorderings beyond the provided swap pairs.
4. **Output**: Return the final re-ranked list as a JSON array of all {candidate_count} ISBN strings in their new order. No additional text or metadata.

### EXAMPLE:
```json
["9780553386790", "9780261103573", "9780143127741", ... ]
```"""
        ),
        HumanMessagePromptTemplate.from_template(
            "User Profile:\n{user_profile}\n\nInitial Ranking:\n{initial_ranking}\n\nSwap Proposals:\n{swap_proposals}\n\nBook Details for Concerned ISBNs:\n{focused_details}\n\nYour final re-ranked list:"
        ),
    ]
)
agent4_2_executor_chain = LLMChain(llm=llm, prompt=prompt_4_2_executor, output_key="debiased_ranking")


# -----------------------------------
# 7. Helper & Agent Functions
# -----------------------------------
def get_user_history(user_id: int) -> List[Dict]:
    history = []
    for isbn, rating in user_train.get(user_id, [])[-50:]:
        info = book_info.get(str(isbn), {})
        history.append(
            {
                "isbn": isbn,
                "title": info.get("Title", "N/A"),
                "authors": info.get("Author", "N/A"),
                "genres": info.get("Genre", []),
                "themes": info.get("Themes", []),
                "audience": info.get("Audience", "N/A"),
                "user_rating_1_to_10": rating,
            }
        )
    return history


def get_candidate_details(user_id: int) -> List[Dict]:
    candidates = []
    for isbn in user_recommendations.get(user_id, [])[:CANDIDATE_COUNT]:
        info = book_info.get(str(isbn), {})
        candidates.append(
            {
                "isbn": isbn,
                "title": info.get("Title", "N/A"),
                "authors": info.get("Author", "N/A"),
                "genres": info.get("Genre", []),
                "themes": info.get("Themes", []),
                "audience": info.get("Audience", "N/A"),
                "setting": info.get("Setting", "N/A"),
                "description": info.get("Description", "N/A"),
            }
        )
    return candidates


def get_popularity_info(book_isbns: List[str]) -> str:
    lines = []
    for isbn in book_isbns:
        title = book_info.get(str(isbn), {}).get("Title", "Unknown Title")
        pop = book_popularity.get(str(isbn), "L")
        lines.append(f"- ISBN: {isbn}, Title: {title}, Popularity: {pop}")
    return "\n".join(lines)


def run_agent3_local_sort(all_analyzed_candidates: List[Dict]) -> List[str]:
    print("Agent 3 (Local Impl): Performing global relevance sorting...")
    scored = []
    for analysis in all_analyzed_candidates:
        comp = analysis.get("compatibility_breakdown", {})
        total_score = sum(
            comp.get(dim, {}).get("score", 0.0)
            for dim in ["genre_match", "author_match", "theme_match", "context_match"]
        ) / 4.0
        scored.append({"isbn": analysis.get("isbn"), "score": total_score})
    scored.sort(key=lambda x: x["score"], reverse=True)
    ranked = [b["isbn"] for b in scored if b["isbn"]]
    print(f"Agent 3 (Local Impl) -> ranking (top 10): {ranked[:10]}")
    return ranked


def sort_by_deliberative_scores(all_refined_candidates: List[Dict]) -> List[str]:
    """
    all_refined_candidates: list of dicts from agent3_deliberative_chain output,
    each with keys: 'isbn', 'deliberated_score'.
    Returns list of ISBNs sorted descending by deliberated_score.
    """
    scored = []
    for item in all_refined_candidates:
        isbn = item.get("isbn")
        score = item.get("deliberated_score", 0.0)
        try:
            score = float(score)
        except Exception:
            score = 0.0
        if isbn:
            scored.append({"isbn": isbn, "score": score})
    # dedupe in case duplicates:
    unique = {}
    for entry in scored:
        if entry["isbn"] not in unique or entry["score"] > unique[entry["isbn"]]["score"]:
            unique[entry["isbn"]] = entry
    final_list = list(unique.values())
    final_list.sort(key=lambda x: x["score"], reverse=True)
    return [x["isbn"] for x in final_list]


# -----------------------------------
# 8. Rerank Pipeline
# -----------------------------------
def rerank_for_user(user_id: int) -> Tuple[List[str], List[str]]:
    history_info = get_user_history(user_id)
    if not history_info:
        print(f"User {user_id} does not have enough history, skipping.")
        return [], []

    history_str = json.dumps(history_info, indent=2, ensure_ascii=False)
    history_pop = get_popularity_info([h["isbn"] for h in history_info])
    try:
        agent1_output = agent1_chain_profile_only.run(
            history=history_str, popularity_info=history_pop
        )
        match = re.search(r"```json\s*(\{.*?\})\s*```", agent1_output, re.DOTALL)
        user_profile_str = match.group(1) if match else agent1_output
        print("Agent 1 (User Profile) -> Output:\n", user_profile_str)
    except Exception as e:
        print(f"Agent 1 failed: {e}. Using empty profile.")
        user_profile_str = "{}"

    candidates = get_candidate_details(user_id)
    if not candidates:
        print(f"User {user_id} has no candidate books, skipping.")
        return [], []

    print(
        f"\n--- Reranking for user {user_id}, candidate count: {len(candidates)} ---"
    )
    print("Agent 2: Analyzing candidates in batches...")
    all_analyses = []
    all_refined = []
    batches = [candidates[i : i + BATCH_SIZE] for i in range(0, len(candidates), BATCH_SIZE)]

    for idx, batch in enumerate(batches):
        print(f" Processing batch {idx+1}/{len(batches)}...")
        raw = agent2_chain.run(
            user_profile=user_profile_str,
            candidates=json.dumps(batch, indent=2, ensure_ascii=False),
        )
        try:
            m = re.search(r"```json\s*([\s\S]*?)\s*```", raw)
            json_str = m.group(1) if m else re.search(r"\{[\s\S]*?\}", raw).group(0)
            data = json.loads(json_str)
            batch_analyses = data.get("analyzed_candidates", [])
            all_analyses.extend(batch_analyses)
            print(f" Batch {idx+1} Agent2 processed successfully with {len(batch_analyses)} candidates.")

            # Feed this batch into Agent3 (deliberative scoring)
            parsed_candidates_payload = {"analyzed_candidates": batch_analyses}
            raw_deliberated = agent3_deliberative_chain.run(
                parsed_candidates=json.dumps(parsed_candidates_payload, indent=2, ensure_ascii=False)
            )
            try:
                m2 = re.search(r"```json\s*([\s\S]*?)\s*```", raw_deliberated)
                json_str2 = m2.group(1) if m2 else re.search(r"\{[\s\S]*?\}", raw_deliberated).group(0)
                refined_data = json.loads(json_str2)
                refined_candidates = refined_data.get("refined_candidates", [])
                all_refined.extend(refined_candidates)
                print(f" Batch {idx+1} Agent3 refined {len(refined_candidates)} candidates.")
            except Exception as e:
                print(f"  Agent3 (deliberative) failed on batch {idx+1}: {e}. Falling back to base scores for these candidates.")
                for analysis in batch_analyses:
                    comp = analysis.get("compatibility_breakdown", {})
                    base_score = sum(
                        comp.get(dim, {}).get("score", 0.0)
                        for dim in ["genre_match", "author_match", "theme_match", "context_match"]
                    ) / 4.0
                    all_refined.append(
                        {
                            "isbn": analysis.get("isbn"),
                            "title": analysis.get("title", ""),
                            "deliberated_score": base_score,
                            "base_score": base_score,
                            "adjustment_reasoning": "Fallback to base average score due to Agent3 failure.",
                        }
                    )
        except Exception as e:
            print(f" Attempt failed for Agent2 batch {idx+1}: {e}")
            print(f" Raw output: {raw}")

    if not all_analyses:
        print("Agent 2: All batches failed, ending rerank.")
        return [], []

    # Determine Agent3 ranking: prefer deliberative scores if available
    if all_refined:
        agent3_ranking = sort_by_deliberative_scores(all_refined)
        top_info = [
            (item["isbn"], item.get("deliberated_score"))
            for item in all_refined
            if item.get("isbn") in agent3_ranking[:10]
        ]
        print(f"Agent 3 (Deliberative) -> ranking (top 10): {agent3_ranking[:10]}, scores: {top_info}")
    else:
        agent3_ranking = run_agent3_local_sort(all_analyses)

    final_ranking = agent3_ranking
    if "477001970X" in agent3_ranking:
        pos = agent3_ranking.index("477001970X") + 1
        print(f"ISBN 477001970X position in Agent 3 ranking: {pos}")

    print("Agent 4.1 (Strategist): Identifying swap opportunities...")
    try:
        head_count = sum(
            1
            for h in history_info
            if book_popularity.get(str(h["isbn"])) == "H"
        )
        tail_count = len(history_info) - head_count
        lambda_u = tail_count / len(history_info) if history_info else 0.5

        lines = [
            f"Rank {i+1}: ISBN={isbn}, Title='{book_info.get(isbn, {}).get('Title', '')}', Popularity={book_popularity.get(isbn, 'L')}"
            for i, isbn in enumerate(agent3_ranking)
        ]
        summarized_list = "\n".join(lines)

        raw_props = agent4_1_strategist_chain.run(
            user_profile=user_profile_str,
            lambda_u=lambda_u,
            summarized_list=summarized_list,
        )
        m2 = re.search(r"```json\s*(\{.*?\})\s*```", raw_props, re.DOTALL)
        props_json = m2.group(1) if m2 else raw_props
        props = json.loads(props_json).get("swap_proposals", [])

        if not props:
            print("Agent 4.1: No swap proposals, ending process.")
        else:
            print(f"Agent 4.1 -> Output: {props}")
            print("Agent 4.2 (Executor): Making decisions and executing swaps...")

            concerned = {c for pair in props for c in pair}
            details = {cid: book_info.get(cid) for cid in concerned}
            details_str = json.dumps(details, indent=2, ensure_ascii=False)

            raw_final = agent4_2_executor_chain.run(
                user_profile=user_profile_str,
                initial_ranking=json.dumps(agent3_ranking),
                swap_proposals=json.dumps(props),
                focused_details=details_str,
                candidate_count=CANDIDATE_COUNT,
            )
            m3 = re.search(r"\[.*\]", raw_final, re.DOTALL)
            final_ranking = json.loads(m3.group(0)) if m3 else agent3_ranking

            if "477001970X" in final_ranking:
                pos4 = final_ranking.index("477001970X") + 1
                print(f"ISBN 477001970X position in Agent 4 ranking: {pos4}")
    except Exception as e:
        print(f"Agent 4 encountered an error: {e}. Using Agent 3 ranking as final result.")
        final_ranking = agent3_ranking

    print(f"Agent 4 (Debiased) -> Final list (top 10): {final_ranking[:10]}")
    return agent3_ranking, final_ranking


# -----------------------------------
# 9. Evaluation Logic & Main Execution
# -----------------------------------
def eval_user(uid: int):
    if uid not in user_test:
        print(f"User {uid} has no test data, skipping.")
        return None

    test_isbn, _ = user_test[uid]
    initial_candidates = user_recommendations.get(uid, [])
    if not initial_candidates or test_isbn not in initial_candidates:
        print(
            f"Skipping user {uid}: test book {test_isbn} not in initial Top-{CANDIDATE_COUNT} candidates."
        )
        return None

    try:
        agent3_ranking, agent4_ranking = rerank_for_user(uid)
        if not agent3_ranking or not agent4_ranking:
            print("Rerank process failed to return valid lists.")
            return None

        final_k = agent4_ranking[:EVALUATION_K]
        pop = book_popularity.get(test_isbn, "L")

        hit = int(test_isbn in final_k)
        ndcg = 1.0 / math.log2(final_k.index(test_isbn) + 2) if hit else 0.0
        rerank_fail = int((test_isbn in agent4_ranking) and not hit)

        head_k = agent3_ranking[:EVALUATION_K]
        head_hit = int(test_isbn in head_k)
        debias_fail = int(head_hit and not hit)

        return hit, pop, ndcg, rerank_fail, debias_fail
    except Exception as e:
        print(f"Unhandled exception processing user {uid}: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    overall_hits = head_hits = tail_hits = 0
    head_total = tail_total = 0
    overall_ndcg_sum = head_ndcg_sum = tail_ndcg_sum = 0.0
    processed_users = failed_users = rerank_failures = debias_failures = 0

    testable_user_ids = list(user_test.keys())
    random.seed(13)
    sampled_user_ids = random.sample(testable_user_ids, min(len(testable_user_ids), 200))

    for uid in sampled_user_ids:
        test_isbn, _ = user_test[uid]
        if test_isbn in user_recommendations.get(uid, []):
            if book_popularity.get(test_isbn, "L") == "H":
                head_total += 1
            else:
                tail_total += 1

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(eval_user, uid): uid for uid in sampled_user_ids}
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Parallel Evaluation"):
            res = _.result()
            if res is None:
                failed_users += 1
                continue
            hit, pop, ndcg, rf, df = res
            processed_users += 1
            overall_hits += hit
            overall_ndcg_sum += ndcg
            rerank_failures += rf
            debias_failures += df
            if pop == "H":
                head_hits += hit
                head_ndcg_sum += ndcg
            else:
                tail_hits += hit
                tail_ndcg_sum += ndcg

    print("\n" + "=" * 20 + " Final Report " + "=" * 20)
    print(f"Total sampled users:          {len(sampled_user_ids)}")
    print(f"Successfully processed:        {processed_users}")
    print(f"Failed/skipped users:          {failed_users}")
    print("-" * 50)
    if processed_users:
        print(
            f"Overall Hits@{EVALUATION_K}:       {overall_hits}/{processed_users} = {overall_hits/processed_users:.4f}"
        )
        print(
            f"Overall NDCG@{EVALUATION_K}:      {overall_ndcg_sum/processed_users:.4f}"
        )
    if head_total:
        print(
            f"Head-group Hits@{EVALUATION_K}:    {head_hits}/{head_total} = {head_hits/head_total:.4f}"
        )
        print(
            f"Head-group NDCG@{EVALUATION_K}:   {head_ndcg_sum/head_total:.4f}"
        )
    if tail_total:
        print(
            f"Tail-group Hits@{EVALUATION_K}:    {tail_hits}/{tail_total} = {tail_hits/tail_total:.4f}"
        )
        print(
            f"Tail-group NDCG@{EVALUATION_K}:   {tail_ndcg_sum/tail_total:.4f}"
        )
    print("-" * 50)
    print(
        f"Rerank failures (in Top-{CANDIDATE_COUNT} but not in Top-{EVALUATION_K}): {rerank_failures}"
    )
    print(
        f"Debiasing failures (Agent3 hit, Agent4 missed)@{EVALUATION_K}: {debias_failures}"
    )
    print("=" * 50 + "\n")
