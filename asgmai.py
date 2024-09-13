# Imports
import streamlit as st
import time
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from youtubesearchpython import VideosSearch
import difflib
import joblib
import os

# Load the dataset
data = pd.read_csv("spotify_songs.csv")

# Select only the required columns
filtered_data = data[['track_name', 'playlist_subgenre', 'valence', 'energy', 'track_artist']]

# Drop rows where 'track_name' or 'track_artist' is NaN
filtered_data = filtered_data.dropna(subset=['track_name', 'track_artist'])
filtered_data['track_name'] = filtered_data['track_name'].astype(str)  # Ensure all track names are strings

# Convert 'playlist_subgenre' to numeric using one-hot encoding
data_encoded = pd.get_dummies(filtered_data['playlist_subgenre'])

# Scale 'valence' and 'energy'
scaler = StandardScaler()
filtered_data.loc[:, 'valence'] = scaler.fit_transform(filtered_data[['valence']])
filtered_data.loc[:, 'energy'] = scaler.fit_transform(filtered_data[['energy']])

# Combine encoded and scaled features
features = pd.concat([data_encoded, filtered_data[['valence', 'energy']]], axis=1)

# Train the k-nearest neighbors model
knn = NearestNeighbors(n_neighbors=10, metric='euclidean')
knn.fit(features)

# Function to find the closest matching song using fuzzy matching
def find_closest_song(song_name, song_list):
    closest_match = difflib.get_close_matches(song_name, song_list, n=1, cutoff=0.6)  # 60% similarity cutoff
    if closest_match:
        return closest_match[0]
    else:
        return None

# Function to search YouTube for a song and return the first video link
def get_youtube_link(song_name, artist_name):
    search_query = f"{song_name} {artist_name} official"
    videos_search = VideosSearch(search_query, limit=1)
    result = videos_search.result()
    
    if result['result']:
        return result['result'][0]['link']  # Return the first YouTube video link
    else:
        return None

# Song recommendation function (recommend only 5 songs)
def recommend_song(song_name, artist_name):
    # Find the closest matching song and artist in the dataset
    closest_song_row = filtered_data[
        (filtered_data['track_name'].str.contains(song_name, case=False)) &
        (filtered_data['track_artist'].str.contains(artist_name, case=False))
    ]
    
    if closest_song_row.empty:
        st.error("No similar songs in the database.")
        return
    else:
        # Extract the closest match details
        closest_song = closest_song_row['track_name'].values[0]
        closest_artist = closest_song_row['track_artist'].values[0]
        st.success(f"Closest match found: '**{closest_song}**' by **{closest_artist}**")
    
    # Extract the features of the closest matching song
    input_features = features[filtered_data['track_name'] == closest_song]
    
    # Find the nearest neighbors
    distances, indices = knn.kneighbors(input_features)
    
    # Retrieve the recommended songs and artists, excluding the input song
    recommendations = filtered_data.iloc[indices[0]][['track_name', 'track_artist']].values
    with st.spinner('Recommending...'):
        time.sleep(3)
    
    st.subheader(f"Songs similar to '**{closest_song}**' by **{closest_artist}**:")
    st.divider()
    
    recommended_songs = set()  # Use a set to avoid duplicates
    count = 0  # Limit to 5 recommendations
    for rec in recommendations:
        if count >= 5:
            break
        song, artist = rec
        if song != closest_song and (song, artist) not in recommended_songs:
            recommended_songs.add((song, artist))
            youtube_link = get_youtube_link(song, artist)
            if youtube_link:
                st.write(f"'**{song}**' by **{artist}**")
                st.write(f"[YouTube Link]({youtube_link})")
                st.divider()
            else:
                st.write(f"'**{song}**' by **{artist}**: No YouTube link found")
                st.divider()
            count += 1

# Function to display top 5 happy and sad songs
def top_songs_by_mood(filtered_data):
    happy_songs = filtered_data[(filtered_data['valence'] > 0.5) & (filtered_data['energy'] > 0.5)]
    sad_songs = filtered_data[filtered_data['valence'] <= 0]
    
    top_happy_songs = happy_songs.sort_values(by=['valence', 'energy'], ascending=False).head(5)
    top_sad_songs = sad_songs.sort_values(by=['valence', 'energy'], ascending=True).head(5)

    st.subheader("Top 5 Happy Songs:")
    for index, row in top_happy_songs.iterrows():
        st.write(f"'{row['track_name']}' by {row['track_artist']}")

    st.subheader("Top 5 Sad Songs:")
    for index, row in top_sad_songs.iterrows():
        st.write(f"'{row['track_name']}' by {row['track_artist']}")

# Streamlit interface
st.title("Recommend Song Based on Mood😊😔📊")
st.write("Feeling a type of mood? We don't judge! Input a song of your choice that matches how you're feeling, and we'll recommend songs that match the mood!")

# Initialize session state for showing/hiding top 5 songs
if 'show_top_songs' not in st.session_state:
    st.session_state.show_top_songs = False

# Button to toggle showing or hiding top 5 songs
if st.session_state.show_top_songs:
    if st.button("Close Top 5 Songs"):
        st.session_state.show_top_songs = False
else:
    if st.button("Show Top 5 Happy and Sad Songs"):
        st.session_state.show_top_songs = True

# Show top 5 songs when the toggle is active
if st.session_state.show_top_songs:
    top_songs_by_mood(filtered_data)

# Get song name and artist name from the user
input_song = st.text_input("🎶Enter the song name:")
input_artist = st.text_input("👩‍🎤Enter the artist name🧑‍🎤:")

# Recommend button functionality
if st.button("Recommend"):
    if input_song and input_artist:
        recommend_song(input_song, input_artist)
    else:
        st.error("Please enter both the song name and the artist name.")
