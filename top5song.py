# Imports
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import difflib  # Import difflib for fuzzy matching
import joblib  # Import joblib for saving and loading recommendations
import os  # Import os to check if the file exists

# Load the dataset
data = pd.read_csv("spotify_songs.csv")

# Select only the required columns
filtered_data = data[['track_name', 'playlist_subgenre', 'energy', 'valence', 'track_artist']]
filtered_data = filtered_data.dropna(subset=['track_name', 'track_artist'])  # Drop rows where 'track_name' is NaN
filtered_data['track_name'] = filtered_data['track_name'].astype(str)  # Ensure all track names are strings

# Scale energy and valence using StandardScaler
scaler = StandardScaler()
filtered_data.loc[:, 'energy'] = scaler.fit_transform(filtered_data[['energy']])
filtered_data.loc[:, 'valence'] = scaler.fit_transform(filtered_data[['valence']])

# Function to save the 10 latest recommendations
def save_recommendations(recommendations, file_name="recommendations.pkl"):
    if os.path.exists(file_name):
        existing_recommendations = joblib.load(file_name)
        recommendations = list(existing_recommendations) + list(recommendations)  # Append new recommendations
    recommendations = list(recommendations)[-10:]  # Convert set to list and keep only the last 10 recommendations
    joblib.dump(recommendations, file_name)  # Save to file

# Function to load and display the latest 10 saved recommendations
def display_saved_recommendations(file_name="recommendations.pkl"):
    if os.path.exists(file_name):
        saved_recommendations = joblib.load(file_name)
        if saved_recommendations:
            print("\nLatest 10 Recommended Songs:")
            for song, artist in saved_recommendations:
                print(f"'{song}' by {artist}")
        else:
            print("No previous recommendations found.")
    else:
        print("No previous recommendations found.")

# Function to recommend 5 songs based on a song name and artist name
def recommend_song(song_name, artist_name):
    closest_song_row = filtered_data[
        (filtered_data['track_name'].str.contains(song_name, case=False)) &
        (filtered_data['track_artist'].str.contains(artist_name, case=False))
    ]
    
    if closest_song_row.empty:
        print(f"No close match found for '{song_name}' by '{artist_name}' in the dataset.")
        return
    else:
        closest_song = closest_song_row['track_name'].values[0]
        closest_artist = closest_song_row['track_artist'].values[0]
        print(f"Closest match found: '{closest_song}' by {closest_artist} \n\n")
    
    input_features = features[filtered_data['track_name'] == closest_song]
    distances, indices = knn.kneighbors(input_features)
    
    recommendations = filtered_data.iloc[indices[0]][['track_name', 'track_artist']].values
    print(f"Songs similar to '{closest_song}' by {closest_artist}:")
    
    recommended_songs = set()  # Use a set to avoid duplicates
    for rec in recommendations[:6]:  # Limit to 5 recommendations
        song, artist = rec
        if song != closest_song and (song, artist) not in recommended_songs:  # Avoid duplicates and the input song
            recommended_songs.add((song, artist))
            print(f"'{song}' by {artist}")
    
    # Save the 5 recommended songs for future display
    save_recommendations(recommended_songs)

# Function to display the top 5 happy and sad songs based on energy and valence
def top_songs_by_mood(filtered_data):
    happy_songs = filtered_data[(filtered_data['valence'] > 0.5) & (filtered_data['energy'] > 0.5)]
    sad_songs = filtered_data[filtered_data['valence'] <= 0]
    
    top_happy_songs = happy_songs.sort_values(by=['valence', 'energy'], ascending=False).head(5)
    top_sad_songs = sad_songs.sort_values(by=['valence', 'energy'], ascending=True).head(5)

    print("\nTop 5 Happy Songs:")
    for index, row in top_happy_songs.iterrows():
        print(f"'{row['track_name']}' by {row['track_artist']} (Valence: {row['valence']}, Energy: {row['energy']})")

    print("\nTop 5 Sad Songs:")
    for index, row in top_sad_songs.iterrows():
        print(f"'{row['track_name']}' by {row['track_artist']} (Valence: {row['valence']}, Energy: {row['energy']})")

# Display the 10 latest recommended songs
display_saved_recommendations()

# Display top 5 happy and sad songs
top_songs_by_mood(filtered_data)

# Get song name and artist name from the user for recommendations
input_song = input("Enter the song name: ")
input_artist = input("Enter the artist name: ")
recommend_song(input_song, input_artist)

# Save filtered data for future reference
filtered_data.to_csv('filtered_spotify_songs.csv', index=False)
print("Filtered data has been saved as 'filtered_spotify_songs.csv'")
