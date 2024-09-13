# Imports
import streamlit as st
import time
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from youtubesearchpython import VideosSearch
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

# Scale 'energy' and 'energy'
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

# Song recommendation function
def recommend_song(song_name, artist_name):
    # Find the closest matching song and artist in the dataset
    closest_song_row = filtered_data[
        (filtered_data['track_name'].str.contains(song_name, case=False)) &
        (filtered_data['track_artist'].str.contains(artist_name, case=False))
    ]
    
    if closest_song_row.empty:
        st.error("No similar songs in database")
        #st.write(f"No close match found for '{song_name}' by '{artist_name}' in the dataset.")
        return
    else:
        # Extract the closest match details
        closest_song = closest_song_row['track_name'].values[0]
        closest_artist = closest_song_row['track_artist'].values[0]
        st.success(f"Closest match found: '**{closest_song}**' by **{closest_artist}**")
        #st.write(f"Closest match found: '{closest_song}' by {closest_artist}")
    
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
    for rec in recommendations:
        song, artist = rec
        if song != closest_song and (song, artist) not in recommended_songs:  # Avoid duplicates and the input song
            recommended_songs.add((song, artist))
            youtube_link = get_youtube_link(song, artist)
            if youtube_link:
                st.write(f"'**{song}**' by **{artist}**")
                st.write(f"[YouTube Link]({youtube_link})")
                st.divider()
            else:
                st.write(f"'**{song}**' by **{artist}**: No YouTube link found")
                st.divider()


# Streamlit interface
st.title("Recommend Song Based on Mood😊😔📊")
st.write("Feeling a type of mood? We dont judge! Input a song of your choice that matches how your feeling and we'll recommend you songs that match the mood of your choice!")

# Get song name and artist name from the user
input_song = st.text_input("🎶Enter the song name:")
input_artist = st.text_input("👩‍🎤Enter the artist name🧑‍🎤:")


if st.button("Recommend"):
    if input_song and input_artist:
        recommend_song(input_song, input_artist)
    else:
        st.error("Please enter both the song name and the artist name.")
