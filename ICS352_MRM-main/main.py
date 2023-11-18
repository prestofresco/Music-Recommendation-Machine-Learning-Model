# Imports ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# Dictionaries ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
command_dic = {
    "help" : "This prints the help commands",
    "search Artists":"Allows you to seach for an Artist within this data set",
    "search Songs":"Search for a song",
    "search Artist Songs":"Search for a song by an artist",
    "list Artist Ratings":"Prints an artist ratings",
    "list Song Ratings":"Pritns a song ratings",
    "clear Artist Ratings":"clears the artist ratings",
    "clear Song Ratings":"clears the songs ratings",
    "rate Artist":"Rate an artist",
    "rate Song":"Rate a song",
    "rate Song id":"Rates a song based on the songID",
    "update Artist Model":"updates the recommended artists",
    "update Song Model":"updates the recommended songs",
    "recArtists":"returns the top 50 recommended artists for the user",
    "recSongs":"Prints the top 50 songs recommended for the user",
    "demo" : "This will update both the models and return the top 50 recommended songs and artist base on user input",
    "exit":"terminates the program"
}
artist_rating_dic = {}
song_rating_dic = {}

# Utility Functions -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# prints commands
def help():
    for i in command_dic:
        print("\t",i,":",command_dic[i])

# Returns True if given artist exists.
def existsArtist(search):
    for index, row in artists_main.iterrows():
        artist = str(row['artist'])
        if(artist.lower()==search.lower()):
            return True
    return False

# Returns True if given song name exists for given artist. Case Sensitive!
def existsSongName(searchArtist,searchSong):
    for index, row in songs_main.iterrows():
        song = row['track_name']
        artist = row['artist_1']
        if(artist.lower()==searchArtist.lower()):
            if(song==searchSong):
                return True
    return False

# Returns True if given songid exists. Case Sensitive!
def existsSongID(search):
    for index, row in songs_main.iterrows():
        songid = row['track_id']
        if(songid==search):
            return True
    return False

# Return all artists matching given string sorted by maximum song popularity.
def searchArtists(search):
    artists = artists_main.copy(deep=True)
    artists = artists.dropna(subset=['artist'])
    artists = artists[artists['artist'].str.contains(search, case=False)]
    artists = artists[['artist','avg_pop']]
    artists = artists.sort_values(by='avg_pop', ascending=False)
    return artists

# Return all songs matching given string sorted by popularity.
def searchSongs(search):
    songs = songs_main.copy(deep=True)
    songs = songs.dropna(subset=['track_name'])
    songs = songs[songs['track_name'].str.contains(search, case=False)]
    songs = songs[['track_id','track_name','artist_1','popularity']]
    songs = songs.sort_values(by=['popularity'], ascending=False)
    return songs

# Return all songs matching given song ID sorted by popularity.
def searchSongID(search):
    songs = songs_main.copy(deep=True)
    songs = songs.dropna(subset=['track_id'])
    songs = songs[songs['track_id'].str.contains(search, case=False)]
    songs = songs[['track_id','track_name','artist_1','popularity']]
    songs = songs.sort_values(by=['popularity'], ascending=False)
    return songs

# Return all songs matching a given artist sorted by popularity.
def searchArtistSongs(search):
    songs = songs_main.copy(deep=True)
    songs = songs.dropna(subset=['artist_1'])
    songs = songs[songs['artist_1'].str.contains(search, case=False)]
    songs = songs[['track_id','track_name','artist_1','popularity']]
    songs = songs.sort_values(by=['popularity'], ascending=False)
    return songs

def updateArtistPred():
    # Create date frame of rated artists
    artists_rated = artists_main.copy().dropna(subset=[user]).sort_values(by=[user], ascending=False)
    # Fit the submodels
    submodel_lin = LinearRegression().fit(artists_rated.loc[:,['avg_pop','avg_duration','avg_temp','avg_energy','avg_danceability','avg_loudness','avg_speechiness','avg_instrumentalness','avg_acousticness','avg_liveness','avg_valence']],artists_rated.loc[:,[user]])
    submodel_tree = DecisionTreeRegressor(max_depth=3).fit(artists_rated.loc[:,['avg_pop','avg_duration','avg_temp','avg_energy','avg_danceability','avg_loudness','avg_speechiness','avg_instrumentalness','avg_acousticness','avg_liveness','avg_valence']],artists_rated.loc[:,[user]])
    # Add columns for prediction of the submodels
    artists_rated['est_lin'] = submodel_lin.predict(artists_rated.loc[:,['avg_pop','avg_duration','avg_temp','avg_energy','avg_danceability','avg_loudness','avg_speechiness','avg_instrumentalness','avg_acousticness','avg_liveness','avg_valence']])
    artists_rated['est_tree'] = submodel_tree.predict(artists_rated.loc[:,['avg_pop','avg_duration','avg_temp','avg_energy','avg_danceability','avg_loudness','avg_speechiness','avg_instrumentalness','avg_acousticness','avg_liveness','avg_valence']])
    # Fit the combining model on the submodel predictions
    model = LinearRegression().fit(artists_rated.loc[:,['est_lin','est_tree']],artists_rated.loc[:,[user]])
    # Copy the artists data frame and apply a predicted rating
    artists_pred = artists_main.copy()
    artists_pred['est_lin'] = submodel_lin.predict(artists_pred.loc[:,['avg_pop','avg_duration','avg_temp','avg_energy','avg_danceability','avg_loudness','avg_speechiness','avg_instrumentalness','avg_acousticness','avg_liveness','avg_valence']])
    artists_pred['est_tree'] = submodel_tree.predict(artists_pred.loc[:,['avg_pop','avg_duration','avg_temp','avg_energy','avg_danceability','avg_loudness','avg_speechiness','avg_instrumentalness','avg_acousticness','avg_liveness','avg_valence']])
    artists_pred['est_rating'] = model.predict(artists_pred.loc[:,['est_lin','est_tree']])
    # sort and print the ratings
    artists_pred = artists_pred.sort_values(by=['est_rating'], ascending=False)
    artists_pred = artists_pred[['artist','est_rating',user]]
    print('Recommended Artist were updated')
    return artists_pred

def updateSongPred():
    # Create date frame of rated artists
    songs_rated = songs_main.copy().dropna(subset=[user]).sort_values(by=[user], ascending=False)
    # Fit the submodels
    submodel_lin = LinearRegression().fit(songs_rated.loc[:,['popularity','tempo','energy','danceability','loudness','speechiness','instrumentalness','acousticness','liveness','valence']],songs_rated.loc[:,[user]])
    submodel_tree = DecisionTreeRegressor(max_depth=4).fit(songs_rated.loc[:,['popularity','tempo','energy','danceability','loudness','speechiness','instrumentalness','acousticness','liveness','valence']],songs_rated.loc[:,[user]])
    # Add columns for prediction of the submodels
    songs_rated['est_lin'] = submodel_lin.predict(songs_rated.loc[:,['popularity','tempo','energy','danceability','loudness','speechiness','instrumentalness','acousticness','liveness','valence']])
    songs_rated['est_tree'] = submodel_tree.predict(songs_rated.loc[:,['popularity','tempo','energy','danceability','loudness','speechiness','instrumentalness','acousticness','liveness','valence']])
    # Fit the combining model on the submodel predictions
    model = LinearRegression().fit(songs_rated.loc[:,['est_lin','est_tree']],songs_rated.loc[:,[user]])
    # Copy the artists data frame and apply a predicted rating
    songs_pred = songs_main.copy()
    songs_pred['est_lin'] = submodel_lin.predict(songs_pred.loc[:,['popularity','tempo','energy','danceability','loudness','speechiness','instrumentalness','acousticness','liveness','valence']])
    songs_pred['est_tree'] = submodel_tree.predict(songs_pred.loc[:,['popularity','tempo','energy','danceability','loudness','speechiness','instrumentalness','acousticness','liveness','valence']])
    songs_pred['est_rating'] = model.predict(songs_pred.loc[:,['est_lin','est_tree']])
    # sort and print the ratings
    songs_pred = songs_pred.sort_values(by=['est_rating'], ascending=False)
    songs_pred = songs_pred[['artist_1','track_name','est_rating',user]]
    print('Recommended songs were updated')
    return songs_pred


# Command Functions -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# top 50 artists matching a search string
def com_search_artists():
    print("\tSearch Artists: ")
    search = get_input("Artist Name> ")
    if existsArtist(search):
        artists = searchArtists(search)
        print(artists.head(50).to_string(index=False))
    else:
        print("Artist not found")
    

# top 50 artists matching a search string
def com_search_songs():
    print("\tSearch Songs: ")
    search = get_input("Song Name> ")
    songs = searchSongs(search)
    if(songs.empty):
        print("No Songs Found")
    else:
        print(songs.head(50).to_string(index=False))

# List all songs by a given primary artist. Case Sensitive!
def com_search_artist_songs():
    print("\tSearch Artist Songs: ")
    search = get_input("Artist Name> ")
    songs = searchArtistSongs(search)
    if(songs.empty):
        print("No Songs Found")
    else:
        print(songs.to_string(index=False))

# List all records in the artist rating dictionary.
def com_list_artist_ratings():
    artists = list(artist_rating_dic.items())
    df = pd.DataFrame(artists, columns=['Artist', 'Rating'])
    print(df.head(50).to_string(index=False))
    print("\n")
    
# List all records in the song rating dictionaries.
def com_list_song_ratings():
    songs = [{'Artist': value[1], 'Song': value[2], 'Rating': value[0]} for key, value in song_rating_dic.items()]
    df = pd.DataFrame(songs)
    df = df.sort_values(by='Rating',ascending=False)
    print(df[['Artist', 'Song', 'Rating']].head(50).to_string(index=False))
    print("\n")

# Clears the artist rating dic
def com_clear_artist_ratings():
    artist_rating_dic.clear()
    print(">>Artist ratings deleted\n")

# Clears the song rating dic
def com_clear_song_ratings():
    song_rating_dic.clear()
    print(">>Song ratings deleted\n")

# allows you to search for an artist and rate them if they exist
def com_rate_artist_name():
    artist = get_input("Artist Name> ")
    if existsArtist(artist):
        while True:
            rating = get_input("Rating(1-10)> ")
            if rating.isdigit() and 0 < int(rating) < 11:
                artist_rating_dic[artist] = rating
                print("Rating Saved")
                break
            print("Please enter a number between 1-10")
    else:
        print("ERROR: Artist not Found")
        
# allows you to search for a song and rate it
def com_rate_song_name():
    user_input = get_input("Artist Name> ")
    if existsArtist(user_input):
        user_input = get_input("\tSong Name> ")
        songs = searchSongs(user_input)
        if len(songs) > 0:
            while len(songs) > 0:  
                print('Is this song you want to rate?\n')
                print(songs[['track_name','artist_1','popularity']].head(1).to_string(index=False))
                user_input = get_input('')
                if user_input == 'y':
                    while True:
                        print("Rating[1-10]: ")
                        user_input = get_input(">> ")
                        if user_input.isdigit() and 0<int(user_input)<11:
                            song_rating_dic[songs['track_id'].head(1).iloc[0]] = (user_input,songs['artist_1'].head(1).iloc[0],songs['track_name'].head(1).iloc[0])
                            return
                else:
                    songs = songs.drop(songs.index[0]) 
    print("This song was not found\n")
# allows you to search for a song ID and rate it
def com_rate_song_id():
    user_input = get_input('SongID> ')
    if existsSongID(user_input):
        song = searchSongID(user_input)
        while True:
            print("Rating[1-10]: ")
            user_input = get_input(">> ")
            if user_input.isdigit() and 0 < int(user_input) < 11:
                song_rating_dic[song['track_id'].head(1).iloc[0]] = (user_input, song['artist_1'].head(1).iloc[0], song['track_name'].head(1).iloc[0])
                return
            print('Please enter a number between 1-10')
    print('SongID was not found')

# Console Functions -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Take a string input from command line
def get_input(prompt):
    while True:
        user_input = input(prompt)
        if user_input.strip():
            return user_input
        else:
            print("Invalid - Enter a string")


# Main --------------------------------------------------------------------------------------------
if __name__ == "__main__":
    
    # Load songs and artists
    songs_main = pd.read_csv("dataset/songs_final.csv")
    artists_main = pd.read_csv("dataset/artists_final_all.csv")

    artist_model = 0
    song_model = 0

    artist_pred = pd.DataFrame()
    song_pred = pd.DataFrame()

    # Select user
    while True:
        while True:
            user_code = get_input("Select User: \n 1:NP \n 2:PS \n 3:PK \n 4:HI \n 5:SVM \n")
            if user_code.isdigit() and 0 < int(user_code) < 7:
                break
            print("please select again")
        user = None
        match int(user_code):
            case 1:
                user = 'rating_np'
            case 2:
                user = 'rating_ps'
            case 3:
                user = 'rating_pk'
            case 4:
                user = 'rating_hi'
            case 5:
                user = 'rating_svm'
        break
   
    # Populate dictionary from user artist ratings
    # Populate dictionary from user songs ratings
    artist_rating_dic = artists_main.copy().dropna(subset=[user]).set_index('artist')[user].to_dict()
    song_rating_dic = songs_main.copy().dropna(subset=[user]).set_index('track_id')[[user, 'artist_1','track_name']].apply(tuple, axis=1).to_dict()
    # main loop,  this takes the users input and will run a command based on it
    while True:
        user_input = get_input("Com> ").lower()
        match user_input:
            # prints the help console
            case "help":
                help()
            # searches for a artist
            case "search artists":
                while True:
                    com_search_artists()
                    user_input = get_input("\tSearch Again? [y/n]:")
                    match user_input:
                        case "n":
                            break
                        case _:
                            pass
            # search for a song                
            case "search songs":
                while True:
                    com_search_songs()
                    user_input = get_input("\tSearch Again? [y/n]:")
                    match user_input:
                        case "n":
                            break
                        case _:
                            pass
            # search for an artists song        
            case "search artist songs":
                while True:
                    com_search_artist_songs()
                    user_input = get_input("\tSearch Again? [y/n]:")
                    match user_input:
                        case "n":
                            break
                        case _:
                            pass
            # view the current artist ratings
            case "list artist ratings":
                com_list_artist_ratings()
            # view the current song ratings
            case "list song ratings":
                com_list_song_ratings()
            # clear the artist ratings dictionary
            case "clear artist ratings":
                print(f"This will clear {user} Artist Ratings, do you wish to proceed? [y/n] ")
                user_input = get_input("")
                if user_input == 'y':
                    com_clear_artist_ratings()
            # clears the song ratings dictionary
            case "clear song ratings":
                print(f"This will clear {user} Artist Ratings, do you wish to proceed? [y/n] ")
                user_input = get_input("")
                if user_input == 'y':
                    com_clear_song_ratings()
            # allows you to search for an artist and rate them
            case "rate artist":
                com_rate_artist_name()
            # rate song based on the name
            case "rate song":
                com_rate_song_name()
            # rate a song based on song ID
            case "rate song id":
                com_rate_song_id()
            # terminates the program
            case "exit":
                print("Exiting Program")
                break
            # updates the artist model with current ratings in dictionary
            case "update artist model":
                artist_pred = updateArtistPred()
            # updates the song model with the current song ratings in dictionary
            case "update song model":
                song_pred =updateSongPred()
            # displays the top 50 recommended artist
            case 'recartists':
                print(artist_pred[['artist','est_rating']].head(50).to_string(index=False),'\n')
            # displays the top 50 recommended songs
            case 'recsongs':
                print(song_pred[['artist_1','track_name','est_rating']].head(50).to_string(index=False),'\n')
            # updates and prints recommended songs and artists
            case 'demo':
                print('--------------------------------------------\n')
                artist_pred = updateArtistPred()
                song_pred = updateSongPred()
                print('--------------------------------------------\n')
                print('Top 50 Recommended Artists:\n')
                print(artist_pred[['artist','est_rating']].head(50).to_string(index=False),'\n')
                print('--------------------------------------------\n')
                print('Top 50 Recommended Songs:\n')
                print(song_pred[['artist_1','track_name','est_rating']].head(50).to_string(index=False),'\n')
            # handles user invalid user input
            case _:
                print("Please select another Command, or enter help")
        
           
