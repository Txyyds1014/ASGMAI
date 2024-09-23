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
filtered_data = data[['track_name', 'playlist_subgenre', 'valence', 'energy', 'track_artist']]

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

# Function to display loading with a progress bar
def show_loading_bar():
    progress_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.03)
        progress_bar.progress(percent_complete + 1)

# Function to display songs in a framed format
def display_songs_in_frame(songs, title, border_color):
    with st.container():
        st.markdown(f"<div style='border: 2px solid {border_color}; padding: 10px; border-radius: 10px;'>", unsafe_allow_html=True)
        st.subheader(title)
        for i, (song, artist) in enumerate(songs):
            st.write(f"{i+1}. '**{song}**' by **{artist}**")
        st.markdown("</div>", unsafe_allow_html=True)

# Function to get top 5 happy and sad songs based on energy, with YouTube links
def show_top_5_happy_and_sad_songs():
    # Show loading bar
    show_loading_bar()

    # Sort by energy for happy and sad songs
    sorted_data_happy = filtered_data.sort_values(by='energy', ascending=False)
    sorted_data_sad = filtered_data.sort_values(by='energy', ascending=True)

    # Filter top 5 happy songs (highest energy), ensuring no duplicates
    top_5_happy_songs = sorted_data_happy.drop_duplicates(subset=['track_name']).head(5)
    happy_songs = list(zip(top_5_happy_songs['track_name'], top_5_happy_songs['track_artist']))

    # Filter top 5 sad songs (lowest energy), ensuring no duplicates with happy songs
    top_5_sad_songs = sorted_data_sad[~sorted_data_sad['track_name'].isin(top_5_happy_songs['track_name'])].drop_duplicates(subset=['track_name']).head(5)
    sad_songs = list(zip(top_5_sad_songs['track_name'], top_5_sad_songs['track_artist']))

    # Display happy songs with YouTube links
    with st.container():
        st.markdown(f"<div style='border: 2px solid #4CAF50; padding: 10px; border-radius: 10px;'>", unsafe_allow_html=True)
        st.subheader("Top 5 Motivation Songs 🎉")
        for i, (song, artist) in enumerate(happy_songs):
            youtube_link = get_youtube_link(song, artist)
            if youtube_link:
                st.write(f"{i+1}. '**{song}**' by **{artist}** - [YouTube Link]({youtube_link})")
            else:
                st.write(f"{i+1}. '**{song}**' by **{artist}** (No YouTube link found)")
        st.markdown("</div>", unsafe_allow_html=True)

    # Display sad songs with YouTube links
    with st.container():
        st.markdown(f"<div style='border: 2px solid #FF6347; padding: 10px; border-radius: 10px;'>", unsafe_allow_html=True)
        st.subheader("Top 5 Demotivation Songs 😢")
        for i, (song, artist) in enumerate(sad_songs):
            youtube_link = get_youtube_link(song, artist)
            if youtube_link:
                st.write(f"{i+1}. '**{song}**' by **{artist}** - [YouTube Link]({youtube_link})")
            else:
                st.write(f"{i+1}. '**{song}**' by **{artist}** (No YouTube link found)")
        st.markdown("</div>", unsafe_allow_html=True)

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
st.title("Recommend Song Based on Mood 😊😔📊")
st.write("Feeling a type of mood? We dont judge! Input a song of your choice that matches how your feeling and we'll recommend you songs that match the mood of your choice!")

# Get song name and artist name from the user
input_song = st.text_input("🎶Enter the song name:")
input_artist = st.text_input("👩‍🎤Enter the artist name🧑‍🎤:")

if st.button("Recommend"):
    if input_song and input_artist:
        recommend_song(input_song, input_artist)
    else:
        st.error("Please enter both the song name and the artist name.")

# Show top 5 happy and sad songs
if st.button("Recommend 5 Motivate and Demotivate Songs"):
    show_top_5_happy_and_sad_songs()
