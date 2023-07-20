# %%
# Importing libraries
from sklearn.model_selection import train_test_split
import time
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
import pickle

from dotenv import load_dotenv
import os
load_dotenv()  # load environment variables from .env file
conda_env_path = os.getenv("CONDA_ENV_PATH")
client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")
# print(conda_env_path)  # this should print the path to your Conda environment
# print(client_id)
# print(client_secret)

# %%
# Credentials for Spotify API
client_credentials_manager = SpotifyClientCredentials(
    client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# %%
# fetch techno playlists from spotify
techno_lists = sp.search(q='techno, rave, future rave',
                         type='playlist', limit=50)
techno_playlists = []
techno_songs = pd.DataFrame(columns=['song_name', 'artist_name', 'danceability', 'energy', 'key', 'loudness',
                            'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'like'])

# %%
# iterate through playlists and print all playlist names
# make a guard for playlists with the same name
for i, playlist in enumerate(techno_lists['playlists']['items']):
    if playlist['name'] in [x['name'] for x in techno_playlists]:
        continue
    print("%4d %s %s" % (i + 1, playlist['uri'], playlist['name']))
    techno_playlists.append(sp.playlist(playlist['uri']))
# %%
# fetch songs from each of the techno playlists and put into a dataframe
duplicate_counter = 0
request_counter = 0
for i, playlist in enumerate(techno_playlists):
    print("%4d %s %s" % (i + 1, playlist['uri'], playlist['name']))
    tracks = playlist['tracks']['items']

    # Gather all track ids from the playlist
    track_ids = []
    for j in range(len(tracks)):
        track = tracks[j]['track']
        # make a guard for NoneType, if none, skip
        if track is None or track['id'] is None:
            # skip
            continue
        track_ids.append(track['id'])

    # Request audio features in batches to avoid rate limit error
    for j in range(0, len(track_ids), 100):  # Spotify API allows up to 100 ids per request
        batch = track_ids[j:j+100]

        # if 2 batches have been requested, sleep for 0.4 seconds
        if request_counter % 2 == 0 and request_counter > 0:
            time.sleep(0.4)
        request_counter += 1

        # get the audio features for all songs in the batch
        features_list = sp.audio_features(batch)

        for k, features in enumerate(features_list):
            if not features:
                continue
            feature_keys = ['danceability', 'energy', 'key', 'loudness', 'mode',
                            'speechiness', 'acousticness', 'instrumentalness',
                            'liveness', 'valence', 'tempo']
            # guard for features being NoneType
            if any(features.get(key) is None for key in feature_keys):
                continue
            # guard for name and artist being NoneType
            if tracks[j+k]['track'] is None or tracks[j+k]['track']['name'] is None or tracks[j+k]['track']['artists'][0]['name'] is None:
                continue
            # get the song name and artist
            song_name = tracks[j+k]['track']['name']
            artist_name = tracks[j+k]['track']['artists'][0]['name']
            # check if song_name with a specific artist_name already exists in the dataframe
            if not techno_songs[(techno_songs['song_name'] == song_name) & (techno_songs['artist_name'] == artist_name)].empty:
                duplicate_counter += 1
                print("Duplicate song found: %s" %
                      song_name + " by " + artist_name)
            else:
                # get the audio features
                danceability = features['danceability']
                energy = features['energy']
                key = features['key']
                loudness = features['loudness']
                mode = features['mode']
                speechiness = features['speechiness']
                acousticness = features['acousticness']
                instrumentalness = features['instrumentalness']
                liveness = features['liveness']
                valence = features['valence']
                tempo = features['tempo']
                # add the song and the features to the dataframe
                techno_songs.loc[len(techno_songs)] = [song_name, artist_name, danceability, energy, key,
                                                       loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, 1]

# %%
# describe the dataframe
techno_songs.describe()

# %%
# Duplicates, Total Songs, Batches requested
print("Number of duplicate songs: %d" % duplicate_counter)
print("Number of songs: %d" % len(techno_songs))
print("Number of batches requested: %d" % request_counter)

# %%
# Genres Outside of the desired (Techno, Rave, Future Rave)
# list of genres to be used in the search query
genres = ["pop", "rock", "jazz", "hip-hop", "classical", "country",  "blues", "metal",
          "reggae", "rnb", "folk", "alternative", "indie", "soul", "funk", "punk", "gospel", "latin"]
# print the genres with a numbering
for i, genre in enumerate(genres):
    print("%4d %s" % (i + 1, genre))

# %%
# Fetching all the songs from the different genres and put it into training and testing dataframes

# initialize empty dataframes for the final training and testing sets
train_songs = pd.DataFrame(columns=['song_name', 'artist_name', 'danceability', 'energy', 'key', 'loudness',
                           'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'like'])
test_songs = pd.DataFrame(columns=['song_name', 'artist_name', 'danceability', 'energy', 'key', 'loudness',
                          'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'like'])

request_counter = 0

# iterate over the genres
for genre in genres:
    # search for playlists of that genre
    playlists = sp.search(q=genre, type='playlist', limit=1)

    genre_songs = pd.DataFrame(columns=['song_name', 'artist_name', 'danceability', 'energy', 'key', 'loudness',
                               'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'like'])

    for i, playlist in enumerate(playlists['playlists']['items']):
        print("%4d %s %s" % (i + 1, playlist['uri'], playlist['name']))
        results = sp.playlist(playlist['uri'])

        # limit the number of songs fetched from each playlist
        for j in range(min(50, len(results['tracks']['items']))):

            if request_counter % 100 == 0 and request_counter > 0:
                time.sleep(0.5)
            request_counter += 1

            # skip if track or track ID is None
            if results['tracks']['items'][j]['track'] is None or results['tracks']['items'][j]['track']['id'] is None:
                continue

            # get audio features for the song
            features = sp.audio_features(
                results['tracks']['items'][j]['track']['id'])

            # skip if features list is empty or contains None as its first element
            if not features or features[0] is None:
                continue

            # get song name and artist name
            song_name = results['tracks']['items'][j]['track']['name']
            artist_name = results['tracks']['items'][j]['track']['artists'][0]['name']

            # check if song already exists in the techno songs dataframe, if not, add it
            if not any((techno_songs['song_name'] == song_name) & (techno_songs['artist_name'] == artist_name)):
                # get audio features
                danceability = features[0]['danceability']
                energy = features[0]['energy']
                key = features[0]['key']
                loudness = features[0]['loudness']
                mode = features[0]['mode']
                speechiness = features[0]['speechiness']
                acousticness = features[0]['acousticness']
                instrumentalness = features[0]['instrumentalness']
                liveness = features[0]['liveness']
                valence = features[0]['valence']
                tempo = features[0]['tempo']

                # assign 'like' value based on genre
                # for simplicity, assign 1 if genre is techno, 0 otherwise
                # like = 1 if genre == "techno" else 0

                # add the song and the features to the dataframe
                genre_songs.loc[len(genre_songs)] = [song_name, artist_name, danceability, energy, key,
                                                     loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, 0]

    # split the songs from this genre into training and testing sets
    genre_train_songs, genre_test_songs = train_test_split(
        genre_songs, test_size=0.4)

    # add the training and testing songs from this genre to the final training and testing sets
    train_songs = pd.concat([train_songs, genre_train_songs])
    test_songs = pd.concat([test_songs, genre_test_songs])
# %%
# Displays dislike songs in both training
# print train_songs with like value 0
print("Training songs with like value 0:")
print(train_songs[train_songs['like'] == 0])
# %%
# Displays dislike songs in  testing sets
# print test_songs with like value 0
print("Testing songs with like value 0:")
print(test_songs[test_songs['like'] == 0])

# %%
# save all the data to csv files
techno_songs.to_csv('techno_songs.csv', index=False)
train_songs.to_csv('train_songs.csv', index=False)
test_songs.to_csv('test_songs.csv', index=False)
# techno_playlists.to_csv('techno_playlists.csv', index=False)

# %%
# add all songs onto 1 csv file called all_songs.csv
all_songs = pd.concat([techno_songs, train_songs, test_songs])
all_songs.to_csv('all_songs.csv', index=False)
