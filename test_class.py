import streamlit as st
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pickle
from sklearn.preprocessing import StandardScaler
import config 

# Title of the app
st.title(":green[_Spotify_] Song Recommender")
st.subheader("_Music for Everyone_ :sunglasses:", divider=True)

# Spotify API Authentication
client_id = 'a6db8fab957a42bea4da6fc433c87a93'  
client_secret = '0b8e4df683934d28ac8e978ffdc7df40'  

client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Load the pre-trained KMeans model
with open('spotify_kmeans_model.pkl', 'rb') as model_file:
    kmeans = pickle.load(model_file)

df_songs = pd.read_csv('df5k2.csv')
# Define the scaler
#scaler = StandardScaler()

# Define expected features
expected_features = [
    'danceability', 
    'energy', 
    'loudness', 
    'speechiness', 
    'acousticness', 
    'instrumentalness', 
    'liveness', 
    'valence', 
    'tempo', 
    'duration_ms'
]

# Function to fetch song features from Spotify
def bring_song (song_name):
    result = sp.search(q=song_name, limit=1, market ="ES") #SEARCH FOR THE RESULT
    id = result["tracks"]["items"][0]["id"] #BRING THE ID
    return id

# Function to classify song into clusters
def classify_song (id):
    features = sp.audio_features(id) #BRING ALL FEATURES
    X=pd.DataFrame(features) #TURN ARRAY INTO DATAFRAME
    X=X[["danceability","energy","loudness","speechiness","acousticness",
    "instrumentalness","liveness","valence","tempo","id","duration_ms"]] #CHANGE THE COLUMN NAMES
    X = X.drop("id", axis=1) #DROP ID
    cluster = kmeans.predict(X) #RUN PREDICTION 
    cluster = cluster[0] #CHANGE ARRAY TO NUMBER
    return cluster

def song_recomender (cluster):
    same_cluster_songs = df_songs.loc[df_songs['cluster'] == cluster] #SEARCH FOR ALL THE SONGS WITH THE SAME CLUSTER
    random_sample = same_cluster_songs.sample(n=3) #SELECT RANDOMLY 1 SONG 
    sample = random_sample["names"]
    return sample

# Input form for song name
song_name = st.text_input("Enter the song:", '')

if song_name:
    features = bring_song(song_name)
    
    if features:
        predicted_cluster = classify_song(features)
        st.write(f"Predicted cluster: {predicted_cluster}")

        # Suggest songs from the same cluster
        st.write("Songs from the same cluster:")

        # Display a list of recommended song names (you can adjust how many songs you want to show)
        random_sample = song_recomender(predicted_cluster)
        st.write(random_sample)  # Showing top 10 songs

    else:   
        st.write("Song not found.")




sentiment_mapping = [":material/thumb_down:", ":material/thumb_up:"]
selected = st.feedback("thumbs")
if selected is not None:
    st.markdown(f"You selected: {sentiment_mapping[selected]}")

