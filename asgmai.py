import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import difflib
import joblib
import os

# Load the dataset
@st.cache
def load_data():
    data = pd.read_csv("spotify_songs.csv")
    return data

# Load data
data = load_data()

# Select required columns
filtered_data = data[['track_name', 'playlist_subgenre', 'energy', 'valence', 'track_artist', 'track_album_release_date']]
filtered_data = filtered_data.dropna(subset=['track_name', 'track_artist'])
filtered_data['track_name'] = filtered_data['track_name'].astype(str)

# One-hot encode playlist_subgenre
data_encoded = pd.get_dummies(filtered_data['playlist_subgenre'])

# Scale energy and valence using StandardScaler
scaler = StandardScaler()
filtered_data.loc[:, 'energy'] = scaler.fit_transform(filtered_data[['energy']])
filtered_data.loc[:, 'valence'] = scaler.fit_transform(filtered_data[['valence']])

# Combine encoded genres with scaled energy and valence into a features DataFrame
features = pd.concat([data_encoded, filtered_data[['energy', 'valence']]], axis=1)

# Initialize the Nearest Neighbors model
knn = NearestNeighbors(n_neighbors=10, metric='euclidean')
knn.fit(features)

# Function to display top 10 latest songs based on release date
def top_latest_songs(filtered_data):
    # Ensure the release date is in datetime format
    filtered_data['track_album_release_date'] = pd.to_datetime(filtered_data['track_album_release_date'], errors='coerce')
    
    # Drop any rows where the date couldn't be converted
    filtered_data = filtered_data.dropna(subset=['track_album_release_date'])
    
    # Sort by the release date in descending order (latest first)
    latest_songs = filtered_data.sort_values(by='track_album_release_date', ascending=False).head(10)

    st.write("Top 10 Latest Songs:")
    for index, row in latest_songs.iterrows():
        st.write(f"'{row['track_name']}' by {row['track_artist']} (Released on {row['track_album_release_date'].date()})")

# Streamlit UI to display the top 10 latest songs
if st.button("Show Top 10 Latest Songs"):
    top_latest_songs(filtered_data)


# Function to save the 10 latest recommendations
def save_recommendations(recommendations, file_name="recommendations.pkl"):
    if os.path.exists(file_name):
        existing_recommendations = joblib.load(file_name)
        recommendations = list(existing_recommendations) + list(recommendations)
    recommendations = list(recommendations)[-10:]
    joblib.dump(recommendations, file_name)

# Function to load and display the latest 10 saved recommendations
def display_saved_recommendations(file_name="recommendations.pkl"):
    if os.path.exists(file_name):
        saved_recommendations = joblib.load(file_name)
        if saved_recommendations:
            st.write("Latest 10 Recommended Songs:")
            for song, artist in saved_recommendations:
                st.write(f"'{song}' by {artist}")
        else:
            st.write("No previous recommendations found.")
    else:
        st.write("No previous recommendations found.")

# Function to recommend 5 songs based on song and artist name
def recommend_song(song_name, artist_name):
    closest_song_row = filtered_data[
        (filtered_data['track_name'].str.contains(song_name, case=False)) &
        (filtered_data['track_artist'].str.contains(artist_name, case=False))
    ]
    
    if closest_song_row.empty:
        st.write(f"No close match found for '{song_name}' by '{artist_name}' in the dataset.")
        return
    else:
        closest_song = closest_song_row['track_name'].values[0]
        closest_artist = closest_song_row['track_artist'].values[0]
        st.write(f"Closest match found: '{closest_song}' by {closest_artist}\n")
    
    input_features = features[filtered_data['track_name'] == closest_song]
    distances, indices = knn.kneighbors(input_features)
    
    recommendations = filtered_data.iloc[indices[0]][['track_name', 'track_artist']].values
    st.write(f"Songs similar to '{closest_song}' by {closest_artist}:")
    
    recommended_songs = set()
    for rec in recommendations[:5]:
        song, artist = rec
        if song != closest_song and (song, artist) not in recommended_songs:
            recommended_songs.add((song, artist))
            st.write(f"'{song}' by {artist}")
    
    # Save the 5 recommended songs
    save_recommendations(recommended_songs)

# Function to display top 5 happy and sad songs
def top_songs_by_mood(filtered_data):
    happy_songs = filtered_data[(filtered_data['valence'] > 0.5) & (filtered_data['energy'] > 0.5)]
    sad_songs = filtered_data[filtered_data['valence'] <= 0]
    
    top_happy_songs = happy_songs.sort_values(by=['valence', 'energy'], ascending=False).head(5)
    top_sad_songs = sad_songs.sort_values(by=['valence', 'energy'], ascending=True).head(5)

    st.write("Top 5 Happy Songs:")
    for index, row in top_happy_songs.iterrows():
        st.write(f"'{row['track_name']}' by {row['track_artist']}")

    st.write("\nTop 5 Sad Songs:")
    for index, row in top_sad_songs.iterrows():
        st.write(f"'{row['track_name']}' by {row['track_artist']}")

# Streamlit UI

st.title("Spotify Song Recommender System")

# Display saved recommendations
display_saved_recommendations()

# Display top happy and sad songs
if st.button("Show Top 5 Happy and Sad Songs"):
    top_songs_by_mood(filtered_data)

# Get song name and artist name from the user
st.subheader("Search for Song Recommendations")
input_song = st.text_input("Enter the song name:")
input_artist = st.text_input("Enter the artist name:")

if st.button("Recommend Songs"):
    recommend_song(input_song, input_artist)

# Save filtered data
filtered_data.to_csv('filtered_spotify_songs.csv', index=False)
st.write("Filtered data has been saved as 'filtered_spotify_songs.csv'")

