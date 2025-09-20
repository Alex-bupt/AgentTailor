import json
import random


# Function to generate movie description text
def movie_template(title, director, actors, genre, description, rating):
    actors_list = ", ".join(actors)
    genre_list = ", ".join(genre)
    movie_info = (f"{title} - User Rating: {rating}, directed by {director}, features an ensemble cast including {actors_list}. It "
                  f"falls under the genres of {genre_list}. The story follows {description}.")
    return movie_info


# Function to generate user profile text
def user_template(user_id, likes, dislikes, other_traits):
    likes_list = ", ".join(likes)
    dislikes_list = ", ".join(dislikes)
    user_info = (f"User {user_id} enjoys movies with elements such as {likes_list}. They tend to dislike films that "
                 f"feature {dislikes_list}. In general, {other_traits}")
    return user_info


# Function to generate task text
def task_template(user_id, movies_info):
    movie_info = "\n".join([f"{movie}" for movie in movies_info])
    task_description = f"""Task: Create a user preference profile for User {user_id} based on the provided movie information and ratings.
    Below are the related movies and their ratings:
    {movie_info}
    Use the movie ratings and information to generate a preference profile for the user, highlighting their preferred genres, actors, directors, and specific movie themes or elements they enjoy.
    """
    return task_description


# Load the movie data from movie.json
with open('./ml-1m/json/movies_filled.json', 'r', encoding="utf-8") as movie_file:
    movie_data = json.load(movie_file)

# Load the user profile data from user_profiles.json
with open('./ml-1m/json/user_profiles.json', 'r', encoding="utf-8") as user_profiles_file:
    user_profiles_data = json.load(user_profiles_file)

# Read ratings data from ratings.txt
ratings_data = []
with open('./ml-1m/ratings.txt', 'r', encoding="utf-8") as ratings_file:
    for line in ratings_file:
        user_id, movie_id, rating, timestamp = line.strip().split('::')
        ratings_data.append((int(user_id), int(movie_id), int(rating), int(timestamp)))

# Extract the last four ratings for each user
user_ratings = {}
for user_id, movie_id, rating, timestamp in ratings_data:
    if user_id not in user_ratings:
        user_ratings[user_id] = []
    user_ratings[user_id].append((movie_id, rating))

# For each user, extract the second to fifth last ratings and get the relevant movie info
user_task_info = []
for user_id, ratings in user_ratings.items():
    last_ratings = ratings[-21:-1]  # Get the ratings from the second to the 21st last
    group_size = 4  # Split the 20 ratings into 5 groups of 4 ratings each

    # Generate 5 task texts for each group of 4 ratings
    for i in range(5):
        group_ratings = last_ratings[i * group_size: (i + 1) * group_size]  # Slice the group
        movies_info = []

        # Loop through the movie_data to find the relevant movie info based on movie_id
        for movie_id, rating in group_ratings:
            movie_info = next((movie for movie in movie_data if int(movie['MovieID']) == movie_id), {})
            movie_info.get("Title", "")
            movie_text = movie_template(movie_info.get("Title", ""), movie_info.get("Director", ""),
                                        movie_info.get("Actors", ""),
                                        movie_info.get("Genre", ""), movie_info.get("Description", ""), rating)
            movies_info.append(movie_text)

        user_info = user_profiles_data.get(str(user_id), {})
        likes = user_info.get("Likes", [])
        dislikes = user_info.get("Dislikes", [])
        other_traits = user_info.get("OtherTraits", "")

        task_text = task_template(user_id, movies_info)
        user_text = user_template(user_id, likes, dislikes, other_traits)

        user_task_info.append({
            "text": [task_text, user_text],
            "task_id": user_id
        })

# Shuffle the user_task_info to create a final dataset
random.shuffle(user_task_info)

# Save the shuffled dataset to a new JSON file
with open('generation.jsonl', 'w', encoding='utf-8') as output_file:
    for entry in user_task_info:
        output_file.write(json.dumps(entry, ensure_ascii=False) + '\n')

# Print out the first few items from the shuffled dataset for verification
for item in user_task_info[:3]:
    print(json.dumps(item, indent=4))

