# Imports
import streamlit as st
import time
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from youtubesearchpython import VideosSearch

# Load the dataset
data = pd.read_csv("spotify_songs.csv")

# Select only the required columns
filtered_data = data[['track_name', 'playlist_subgenre', 'valence', 'energy', 'track_artist', 'track_popularity']]

# Drop rows where 'track_name' or 'track_artist' is NaN
filtered_data = filtered_data.dropna(subset=['track_name', 'track_artist'])
filtered_data['track_name'] = filtered_data['track_name'].astype(str)  # Ensure all track names are strings

# Convert 'playlist_subgenre' to numeric using one-hot encoding
data_encoded = pd.get_dummies(filtered_data['playlist_subgenre'])

# Scale 'energy' and 'valence'
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

# Function to display loading with a progress bar
def show_loading_bar():
    progress_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.03)
        progress_bar.progress(percent_complete + 1)

# Function to display songs in a framed format with YouTube links
def display_songs_in_frame(songs, title, border_color):
    with st.container():
        st.markdown(f"<div style='border: 2px solid {border_color}; padding: 10px; border-radius: 10px;'>", unsafe_allow_html=True)
        st.subheader(title)
        for i, (song, artist, popularity) in enumerate(songs):
            st.write(f"{i+1}. '**{song}**' by **{artist}** (Popularity: {popularity})")
            youtube_link = get_youtube_link(song, artist)
            if youtube_link:
                st.write(f"[YouTube Link]({youtube_link})")
            else:
                st.write("No YouTube link found")
            st.divider()
        st.markdown("</div>", unsafe_allow_html=True)

# Function to get top 5 happy and sad songs with YouTube links
def show_top_5_happy_and_sad_songs():
    # Show loading bar
    show_loading_bar()
    
    # Filter songs with popularity between 80 and 100
    popular_songs = filtered_data[(filtered_data['track_popularity'] >= 70) & (filtered_data['track_popularity'] <= 100)]
    
    # Filter top 10 happy songs (high valence, high energy) from the popular songs and sort by popularity
    top_10_happy_songs = popular_songs.sort_values(by=['valence', 'energy', 'track_popularity'], ascending=[False, False, False]).head(10)
    happy_songs = list(zip(top_10_happy_songs['track_name'], top_10_happy_songs['track_artist'], top_10_happy_songs['track_popularity']))  # Include popularity

    # Filter top 10 sad songs (low valence, low energy) from the popular songs and sort by popularity
    top_10_sad_songs = popular_songs.sort_values(by=['valence', 'energy', 'track_popularity'], ascending=[True, True, False]).head(10)
    sad_songs = list(zip(top_10_sad_songs['track_name'], top_10_sad_songs['track_artist'], top_10_sad_songs['track_popularity']))  # Include popularity
    
    # Ensure unique recommendations and limit to 5
    unique_happy_songs = []
    unique_sad_songs = []
    
    for song in happy_songs:
        if len(unique_happy_songs) < 5 and song not in unique_happy_songs:
            unique_happy_songs.append(song)
            
    for song in sad_songs:
        if len(unique_sad_songs) < 5 and song not in unique_sad_songs:
            unique_sad_songs.append(song)
    
    # In case there are fewer than 5 unique songs, fill up to 5 with available songs
    while len(unique_happy_songs) < 5 and len(happy_songs) > len(unique_happy_songs):
        for song in happy_songs:
            if len(unique_happy_songs) < 5 and song not in unique_happy_songs:
                unique_happy_songs.append(song)
    
    while len(unique_sad_songs) < 5 and len(sad_songs) > len(unique_sad_songs):
        for song in sad_songs:
            if len(unique_sad_songs) < 5 and song not in unique_sad_songs:
                unique_sad_songs.append(song)
    
    # Display happy and sad songs in a tidy frame
    display_songs_in_frame(unique_happy_songs, "Top 5 Happy Songs 🎉", "#4CAF50")
    display_songs_in_frame(unique_sad_songs, "Top 5 Sad Songs 😢", "#FF6347")


# Song recommendation function
def recommend_song(song_name, artist_name):
    # Find the closest matching song and artist in the dataset
    closest_song_row = filtered_data[
        (filtered_data['track_name'].str.contains(song_name, case=False)) &
        (filtered_data['track_artist'].str.contains(artist_name, case=False))
    ]
    
    if closest_song_row.empty:
        st.error("No similar songs in database")
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
    count = 0  # Track the number of displayed recommendations
    for rec in recommendations:
        song, artist = rec
        if song != closest_song and (song, artist) not in recommended_songs and count < 5:  # Limit to 5 songs
            recommended_songs.add((song, artist))
            youtube_link = get_youtube_link(song, artist)
            if youtube_link:
                st.write(f"'**{song}**' by **{artist}**")
                st.write(f"[YouTube Link]({youtube_link})")
            else:
                st.write(f"'**{song}**' by **{artist}**: No YouTube link found")
            st.divider()
            count += 1  # Increment count

# Streamlit interface
st.title("Recommend Song Based on Mood 😊😔📊")
st.write("Feeling a type of mood? Input a song of your choice and we'll recommend similar songs that match the mood!")

# Get song name and artist name from the user
input_song = st.text_input("🎶Enter the song name:")
input_artist = st.text_input("👩‍🎤Enter the artist name🧑‍🎤:")

if st.button("Recommend"):
    if input_song and input_artist:
        recommend_song(input_song, input_artist)
    else:
        st.error("Please enter both the song name and the artist name.")

# Show top 5 happy and sad songs
if st.button("Show Top 5 Happy and Sad Songs"):
    show_top_5_happy_and_sad_songs()
