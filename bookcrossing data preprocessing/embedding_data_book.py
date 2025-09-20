import json
import random

# ----------------------------
# Template functions
# ----------------------------

def convert_user_profile_full_template(user_profile):
    """
    Generate a plain-text representation of the user profile,
    with tagged keywords highlighted and recommendation notes.
    The result will be wrapped as ["Represent this text:", <returned_text>].
    """
    likes = user_profile.get("Likes", [])
    dislikes = user_profile.get("Dislikes", [])
    other = user_profile.get("OtherTraits", "")
    likes_hl = ";".join([f"[{kw}]" for kw in likes]) if likes else "None"
    dislikes_hl = ";".join([f"[{kw}]" for kw in dislikes]) if dislikes else "None"

    text = (
        "<User>\n"
        f"Likes: {likes_hl}\n"
        f"Dislikes: {dislikes_hl}\n"
        f"OtherTraits: {other}\n"
        "</User>"
    )
    return text


def convert_book_info_full_template(book_info, is_pos=True):
    """
    Generate a plain-text representation of book information,
    with tagged keywords highlighted and sample polarity notes.
    is_pos=True indicates a positive sample, otherwise negative.
    """
    author = book_info.get("Author", "")
    year = book_info.get("Year", "")
    genre = book_info.get("Genre", [])
    audience = book_info.get("Audience", "")
    setting = book_info.get("Setting", "")
    themes = book_info.get("Themes", [])
    desc = book_info.get("Description", "")

    author_hl = f"[{author}]" if author else "None"
    year_hl = f"[{year}]" if year else "None"
    genre_hl = ";".join([f"[{g}]" for g in genre]) if genre else "None"
    audience_hl = f"[{audience}]" if audience else "None"
    setting_hl = f"[{setting}]" if setting else "None"
    themes_hl = ";".join([f"[{t}]" for t in themes]) if themes else "None"

    polarity = "Positive Sample" if is_pos else "Negative Sample"
    text = (
        f"<Item ({polarity})>\n"
        f"Author: {author_hl}\n"
        f"Year: {year_hl}\n"
        f"Genre: {genre_hl}\n"
        f"Audience: {audience_hl}\n"
        f"Setting: {setting_hl}\n"
        f"Themes: {themes_hl}\n"
        f"Description: {desc}\n"
        "</Item>"
    )
    return text

# ----------------------------
# Data loading
# ----------------------------

def load_data():
    """
    Read training.json, user.json, and books.json files.
    Construct:
      - output_samples: list of {"user_id": str, "pos": str, "neg": [str,...]}
      - user_profiles: dict mapping user_id to profile dict
      - books_dict: dict mapping ISBN to book info dict (without Title)
    """
    with open("./BX-CSV-Dump/book_training.json", "r", encoding="utf-8") as f:
        output_samples = json.load(f)

    with open("./BX-CSV-Dump/book_user_profiles.json", "r", encoding="utf-8") as f:
        user_profiles = json.load(f)

    with open("./BX-CSV-Dump/books.json", "r", encoding="utf-8") as f:
        books_list = json.load(f)

    # Build ISBN to book info mapping, excluding the Title field
    books_dict = {}
    for book in books_list:
        info = {k: v for k, v in book.items() if k != "Title"}
        books_dict[book["ISBN"]] = info

    return output_samples, user_profiles, books_dict

# ----------------------------
# Sample construction
# ----------------------------

def create_training_samples_with_new_template(output_samples, user_profiles, books_dict):
    """
    Construct training samples using the new template.
    Each sample includes:
      - query: ["Represent this User Profile for Book Retrieval:", user_text]
      - pos: [["Represent this Book:", pos_text]]
      - neg: [["Represent this Book:", neg_text], ...]
      - task_id: fixed to 1
    """
    training_samples = []

    for sample in output_samples:
        user_id = str(sample["user_id"])
        pos_isbn = sample["pos"]
        neg_isbns = sample["neg"]

        user_profile = user_profiles.get(user_id, {})
        pos_book_info = books_dict.get(pos_isbn, {})
        neg_book_infos = [books_dict.get(isbn, {}) for isbn in neg_isbns]

        # Generate text via templates
        user_text = convert_user_profile_full_template(user_profile)
        pos_text = convert_book_info_full_template(pos_book_info, is_pos=True)
        neg_texts = [
            convert_book_info_full_template(info, is_pos=False)
            for info in neg_book_infos
        ]

        # Wrap in gritlm pair format
        query = ["Represent this User Profile for Book Retrieval:", user_text]
        pos_pair = ["Represent this Book:", pos_text]
        neg_pairs = [["Represent this Book:", nt] for nt in neg_texts]

        training_samples.append({
            "query": query,
            "pos": [pos_pair],
            "neg": neg_pairs,
            "task_id": 1
        })

    return training_samples


def write_training_samples(training_samples, output_file):
    """
    Write each training sample as a JSON line to the specified file.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in training_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

# ----------------------------
# Main process
# ----------------------------

def main():
    random.seed(42)

    output_samples, user_profiles, books_dict = load_data()
    training_samples = create_training_samples_with_new_template(
        output_samples, user_profiles, books_dict
    )
    random.shuffle(training_samples)

    write_training_samples(training_samples, "book_embedding.jsonl")
    print(f"Generated {len(training_samples)} training samples and wrote to book_embedding.jsonl")

if __name__ == "__main__":
    main()
