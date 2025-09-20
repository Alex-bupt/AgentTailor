import json
import random

# ----------------------------
# Template functions
# ----------------------------

def book_template(isbn, title, author, year, genre, description, rating):
    """
    Generate a book description text.
    rating: 5 indicates a positive sample (liked), 1 indicates a negative sample (disliked).
    """
    genre_list = ", ".join(genre) if genre else "Unknown"
    return (
        f"{title} (ISBN: {isbn}) - Implied Rating: {rating}/5, by {author} ({year}). "
        f"Genres: {genre_list}. Summary: {description}"
    )


def user_template(user_id, likes, dislikes, other_traits):
    """
    Generate user profile text.
    """
    likes_list = ", ".join(likes) if likes else "None"
    dislikes_list = ", ".join(dislikes) if dislikes else "None"
    return (
        f"User {user_id} enjoys themes such as {likes_list}. "
        f"They tend to dislike {dislikes_list}. {other_traits}"
    )


def task_template(user_id, books_info):
    """
    Generate a task description text, inserting multiple book descriptions.
    """
    book_lines = "\n".join(f"- {b}" for b in books_info)
    return (
        f"Task: Based on the following books and their implied ratings for User {user_id}, "
        "create a concise preference profile highlighting the user's favored authors, genres, themes, "
        "and any patterns you observe in their tastes:\n\n"
        f"{book_lines}"
    )

# ----------------------------
# Data loading
# ----------------------------

with open("./BX-CSV-Dump/book_training.json", "r", encoding="utf-8") as f:
    samples = json.load(f)

with open("./BX-CSV-Dump/books.json", "r", encoding="utf-8") as f:
    books_list = json.load(f)

with open("./BX-CSV-Dump/book_user_profiles.json", "r", encoding="utf-8") as f:
    user_profiles = json.load(f)

# Build mapping from ISBN to book information
books_dict = {book["ISBN"]: book for book in books_list}

# ----------------------------
# Generate generation.jsonl
# ----------------------------

user_task_info = []

for sample in samples:
    user_id = str(sample["user_id"])
    pos_isbn = sample["pos"]
    neg_isbns = sample["neg"]

    # Positive sample book (implied 5-star)
    pos_book = books_dict.get(pos_isbn, {})
    pos_text = book_template(
        pos_isbn,
        pos_book.get("Title", "Unknown Title"),
        pos_book.get("Author", "Unknown Author"),
        pos_book.get("Year", "Unknown Year"),
        pos_book.get("Genre", []),
        pos_book.get("Description", ""),
        rating=5
    )

    # Negative sample books (implied 1-star)
    neg_texts = []
    for isbn in neg_isbns:
        bk = books_dict.get(isbn, {})
        neg_texts.append(
            book_template(
                isbn,
                bk.get("Title", "Unknown Title"),
                bk.get("Author", "Unknown Author"),
                bk.get("Year", "Unknown Year"),
                bk.get("Genre", []),
                bk.get("Description", ""),
                rating=1
            )
        )

    # Combine all book descriptions
    all_books_info = [pos_text] + neg_texts

    # User profile description
    profile = user_profiles.get(user_id, {})
    user_text = user_template(
        user_id,
        profile.get("Likes", []),
        profile.get("Dislikes", []),
        profile.get("OtherTraits", "")
    )

    # Create task entry
    task_text = task_template(user_id, all_books_info)

    user_task_info.append({
        "text": [task_text, user_text],
        "task_id": user_id
    })

# Shuffle and write to JSONL
random.seed(42)
random.shuffle(user_task_info)

with open("generation.jsonl", "w", encoding="utf-8") as out_f:
    for entry in user_task_info:
        out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Generated {len(user_task_info)} generation tasks and wrote to generation.jsonl")
