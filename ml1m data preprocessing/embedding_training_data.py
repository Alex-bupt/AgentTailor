import json
import random


def convert_user_profile_full_template(user_profile):
    """
    Generate plain-text representation of a user profile, with highlighted tags,
    keyword brackets, and recommendation notes. The returned text is used as the second
    element in ["Represent this text:", <returned_text>].
    """
    likes = user_profile.get("Likes", [])
    dislikes = user_profile.get("Dislikes", [])
    other = user_profile.get("OtherTraits", "")

    # Highlight each keyword with square brackets
    likes_hl = ";".join([f"[{kw}]" for kw in likes]) if likes else "None"
    dislikes_hl = ";".join([f"[{kw}]" for kw in dislikes]) if dislikes else "None"

    # Compose multi-line text
    text = (
        "<User>\n"
        f"Likes: {likes_hl}\n"
        f"Dislikes: {dislikes_hl}\n"
        f"OtherTraits: {other}\n"
        "</User>"
    )
    return text


def convert_movie_info_full_template(movie_info, is_pos=True):
    """
    Generate plain-text representation of a movie item, with highlighted tags,
    keyword brackets, and positive/negative sample note.
    is_pos=True indicates a positive sample, otherwise negative.
    """
    actors = movie_info.get("Actors", [])
    director = movie_info.get("Director", "")
    genre = movie_info.get("Genre", [])
    desc = movie_info.get("Description", "")

    # Highlight metadata elements
    actors_hl = ";".join([f"[{a}]" for a in actors]) if actors else "None"
    genre_hl = ";".join([f"[{g}]" for g in genre]) if genre else "None"
    director_hl = f"[{director}]" if director else "None"

    text = (
        "<Item>\n"
        f"Actors: {actors_hl}\n"
        f"Director: {director_hl}\n"
        f"Genre: {genre_hl}\n"
        f"Description: {desc}\n"
        "</Item>"
    )
    return text


def load_data():
    """
    Load training samples, user profiles, and movie details from JSON files.
    Also build a dictionary mapping string movie IDs to movie info with "Title" removed.
    Returns:
        output_samples: list of sample dicts
        user_profiles: dict mapping user_id strings to profile dicts
        movies_dict: dict mapping MovieID strings to movie info dicts (no Title)
    """
    # Load previously generated training samples
    with open("./ml-1m/json/training_data.json", "r", encoding="utf-8") as f:
        output_samples = json.load(f)

    # Load user profiles
    with open("./ml-1m/json/user_profiles.json", "r", encoding="utf-8") as f:
        user_profiles = json.load(f)

    # Load movie metadata list
    with open("./ml-1m/json/movies_filled.json", "r", encoding="utf-8") as f:
        movies_list = json.load(f)

    # Build movie info dictionary, dropping the "Title" field
    movies_dict = {}
    for movie in movies_list:
        # Copy info except Title
        info = {k: v for k, v in movie.items() if k != "Title"}
        movies_dict[movie["MovieID"]] = info

    return output_samples, user_profiles, movies_dict


def create_training_samples_with_new_template(output_samples, user_profiles, movies_dict):
    """
    Construct training samples for embedding, wrapping each sample in the new text templates.
    Each sample will include:
      - query: user profile representation
      - pos: list of positive movie representations
      - neg: list of negative movie representations
      - task_id: static value 1
    """
    training_samples = []
    for sample in output_samples:
        user_id = sample["user_id"]
        pos_movie_id = sample["pos"]
        neg_movie_ids = sample["neg"]

        # Retrieve user profile and movie info
        user_profile = user_profiles.get(str(user_id), {})
        pos_movie_info = movies_dict.get(str(pos_movie_id), {})
        neg_info_list = [movies_dict.get(str(n), {}) for n in neg_movie_ids]

        # Generate text blocks using the new templates
        user_profile_text = convert_user_profile_full_template(user_profile)
        pos_movie_text = convert_movie_info_full_template(pos_movie_info, is_pos=True)
        neg_movie_texts = [convert_movie_info_full_template(mi, is_pos=False) for mi in neg_info_list]

        # Wrap in the required pair format
        query_pair = ["Represent this User Profile for Movie Retrieval:", user_profile_text]
        pos_pairs = [["Represent this Movie:", pos_movie_text]]
        neg_pairs = [["Represent this Movie:", t] for t in neg_movie_texts]

        # Assemble the sample
        training_sample = {
            "query": query_pair,
            "pos": pos_pairs,
            "neg": neg_pairs,
            "task_id": 1
        }
        training_samples.append(training_sample)

    return training_samples


def write_training_samples(training_samples, output_file):
    """
    Write each training sample as a JSON line in the specified file.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in training_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main():
    # Load data and metadata
    output_samples, user_profiles, movies_dict = load_data()

    # Create training samples using the new template
    training_samples = create_training_samples_with_new_template(
        output_samples, user_profiles, movies_dict
    )

    # Shuffle for randomness
    random.shuffle(training_samples)

    # Write to JSONL file
    write_training_samples(training_samples, "embedding.jsonl")
    print(f"Generated {len(training_samples)} training samples and saved to embedding.jsonl.")


if __name__ == "__main__":
    main()
