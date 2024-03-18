# Visualization and plotting imports
import plotly.graph_objs as go
import plotly.express as px

# Basic libraries
import os
import base64
import requests
from requests import post, get
import json
import re

# Data manipulation and analysis imports
import numpy as np
from numpy.linalg import norm
from scipy import stats
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

def get_tdidf(dataset, genre_column):
    tfidf = TfidfVectorizer(token_pattern=r'\b[^\s]+\b')
    tfidf_matrix = tfidf.fit_transform(dataset[genre_column].apply(lambda x: " ".join(x)))
    genre_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())

    return genre_tfidf

def extrair_codigo_playlist(url):
    # Padroniza a regex para encontrar o código da playlist
    padrao = r'/playlist/(\w+)\??'
    
    # Encontra todas as correspondências do padrão na URL
    correspondencias = re.findall(padrao, url)
    
    # Verifica se há correspondências e retorna o código da playlist
    if correspondencias:
        return correspondencias[0]
    else:
        try:
        # Tenta obter a localização do cabeçalho da resposta HTTP
            response = requests.head(url)
            url2 = response.headers.get('location', None)
        
            if url2:
                correspondencias = re.findall(padrao, url2)
                if correspondencias:
                    return correspondencias[0]
        
        # Se não houver correspondências, retorna None
            return None
    
        except requests.exceptions.RequestException as e:
            # Trata a exceção em caso de erro na requisição
            return None
# Spotify API related functions

def get_token(client_id, client_secret):
    auth_string = client_id + ":" + client_secret
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")
    
    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization":"Basic " + auth_base64,
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "client_credentials"}
    result = post(url, headers=headers, data=data)
    json_result = json.loads(result.content)
    token = json_result["access_token"]
    return token

def get_auth_header(token):
    return {"Authorization": "Bearer " + token}

def get_tracks(playlist_url, headers):
    playlist_id = extrair_codigo_playlist(playlist_url)
    url = f'https://api.spotify.com/v1/playlists/{playlist_id}/tracks?limit=99&offset=0'

    response = get(url=url, headers=headers)
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        return "Erro! O link da playlist não é válido."
    
    data = response.json()
    
    tracks_data = data['items']
    tracks = pd.DataFrame({
        'artist': [item['track']['artists'][0]['name'] for item in tracks_data],
        'artist_id': [item['track']['artists'][0]['id'] for item in tracks_data],
        'album_image': [item['track']['album']['images'][0]['url'] for item in tracks_data],
        'track_title': [item['track']['name'] for item in tracks_data],
        'track_href': [item['track']['external_urls']['spotify'].split('/')[-1] for item in tracks_data]
    })
    return tracks

def get_audio_features(href_df, headers):
    url = f'https://api.spotify.com/v1/audio-features?ids='+','.join(href_df)

    response = get(url=url, headers=headers)
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        return "429"
    audio_features = response.json()
    return audio_features

def get_genres(artistIds, headers):
    genres_list = []
    for i in range(0, len(artistIds), 49):
        chunk = artistIds[i:i + 49]
        url_artistId = f'https://api.spotify.com/v1/artists?ids='+','.join(chunk)
        response = get(url=url_artistId, headers=headers)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            return "Erro em get_genres"
        
        artists = response.json()
        genres_list.append([artist['genres'] for artist in artists['artists']])

    genres = [item for sublist in genres_list for item in sublist]
    return pd.DataFrame({'artist_id':artistIds, 'genres':genres})

#def get_audio_features_parallel(track_hrefs, headers):
#    with ThreadPoolExecutor(max_workers=10) as executor:
#        audio_features = list(executor.map(lambda h: get_audio_features(h, headers), track_hrefs))
#    return audio_features

def get_final_dataset(playlist_id, headers):
    tracks = get_tracks(playlist_id, headers)
    if isinstance(tracks, str):
        return tracks
    
    track_hrefs = tracks['track_href'].values
    audio_features = get_audio_features(track_hrefs, headers)
    if isinstance(audio_features, dict):
        final_dataset = pd.concat([tracks, pd.DataFrame(audio_features['audio_features']).drop("track_href", axis=1)], axis=1)
        return final_dataset
    
    return audio_features[0]


# Data manipulation and scaling functions
def scale_columns(final_dataset):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    
    columns_to_scale = ['valence', 'energy']
    
    scaled_data = np.round(scaler.fit_transform(final_dataset[columns_to_scale]),2)
    scaled_df = pd.DataFrame(scaled_data, columns=[f'{col}_scaled' for col in columns_to_scale])
    
    final_dataset = pd.concat([final_dataset, scaled_df], axis=1)
    
    return final_dataset, scaler

def quadrant(x, y):
    if x >= 0 and y >= 0:
        return [[x,y],"Q1", "#FFCA3A"]
    elif x < 0 and y >= 0:
        return [[x,y],"Q2", "#FF595E"]
    elif x < 0 and y < 0:
        return [[x,y],"Q3", "#1982C4"]
    elif x >= 0 and y < 0:
        return [[x,y],"Q4", "#8AC926"]
    else:
        return [[x,y], "Neutro", "#160C28"]

def calculate_metrics(final_dataset):
    columns_scaled = ['valence_scaled', 'energy_scaled']
    
    median = np.round(np.median(final_dataset[columns_scaled], axis=0),2)
    playlist_mood = quadrant(median[0], median[1])
    
    # Calculate score
    A = np.array(final_dataset[columns_scaled])
    final_dataset['cos_similarity'] = 1 - np.round(np.dot(A,median)/(norm(A, axis=1)*norm(median)), 2)
    final_dataset['distance'] = np.round(norm(A - median, axis=1), 2)
    final_dataset['score'] = (final_dataset['distance']) * (final_dataset['cos_similarity']) * (-1)
                                                     
    # Calculate outliers
    outlier_fraction = 0.05
    threshold = stats.scoreatpercentile(final_dataset['score'], 100 * outlier_fraction)
    final_dataset['outlier'] = np.where(final_dataset['score'] < threshold, 1, 0)
    
    return final_dataset, playlist_mood, median

def db_metrics(db_dataset, median, playlist_genres):
    columns_scaled = ['valence_scaled', 'energy_scaled']
    genres = list(playlist_genres.iloc[:,2:].columns)
    median = np.append(median, playlist_genres.iloc[:,2:].values)

    db_dataset = db_dataset.drop_duplicates(subset=['artist_id', 'track_href']).reset_index(drop=True)
    A = np.array(db_dataset[columns_scaled+genres])
    
    db_dataset['cos_similarity'] = 1 - np.round(np.dot(A,median)/(norm(A, axis=1)*norm(median)), 2)
    db_dataset['distance'] = np.round(norm(A - median, axis=1), 2)
    db_dataset['score'] = (db_dataset['distance']) * (db_dataset['cos_similarity']) * (-1)

    inlier_fraction = 0.05
    threshold = stats.scoreatpercentile(db_dataset['score'], 100 * inlier_fraction)
    db_dataset['inlier'] = np.where(db_dataset['score'] >= threshold, 1, 0)

    return db_dataset


# Visualization functions
def create_mood_scatter_plot(playlist_mood, outlier_data, inlier_data):
    # Create scatter plot
    fig = px.scatter(x=[playlist_mood[0][0]], y=[playlist_mood[0][1]], labels={"x": "Valência", "y": "Energia"},
                     template="simple_white")
    
    ## Update layout
    fig.update_layout(
        yaxis_range=[-1.1, 1.1],
        xaxis_range=[-1.1, 1.1],
        yaxis=dict(visible=True, showticklabels=False),
        xaxis=dict(visible=True, showticklabels=False)
    )

    ## Add arrows
    x_end = [0.95, 0]
    y_end = [0, 0.95]
    x_start = [-0.95, 0]
    y_start = [0, -0.95]
    list_of_all_arrows = []

    for x0, y0, x1, y1 in zip(x_end, y_end, x_start, y_start):
        arrow = go.layout.Annotation(dict(
            x=x0,
            y=y0,
            xref='x', yref='y',
            axref='x', ayref='y',
            showarrow=True,
            ax=x1,
            ay=y1,
            arrowhead=1,
            arrowsize=2,
            arrowwidth=1.5,
            arrowcolor="#636363"
        ))

        list_of_all_arrows.append(arrow)

    fig.update_layout(
        annotations=list_of_all_arrows,
        font_family="Mukta Vaani",
        xaxis_title="VALÊNCIA (→)",
        yaxis_title="ENERGIA (→)",
        font=dict(size=16),
        margin=dict(l=5, r=5, t=5, b=10)
    )

    # Add text annotations
    x = [0, 0, -0.95, 0.95]
    y = [0.95, -0.95, 0, 0]
    xshift = [15, 15, 25, -25]
    yshift = [-25, 25, 15, 15]
    text = ["<i>Alta</i>", "<i>Baixa</i>", "<i>Negativa</i>", "<i>Positiva</i>"]

    for x_, y_, t_, xshift_, yshift_ in zip(x, y, text, xshift, yshift):
        fig.add_annotation(x=x_, y=y_,
                           text=t_,
                           showarrow=False,
                           yshift=yshift_,
                           xshift=xshift_,
                           font={"size": 12})

    x = [0.5, -0.5, -0.5, 0.5]
    y = [0.5, -0.5, 0.5, -0.5]
    text = ["<b>Felicidade</b>", "<b>Tristeza</b>", "<b>Tensão</b>", "<b>Tranquilidade</b>"]

    for x_, y_, t_ in zip(x, y, text):
        fig.add_annotation(x=x_, y=y_,
                           text=t_,
                           showarrow=False,
                           font={"size": 12},
                           align="center",
                           opacity=0.25)

    fig.update_xaxes(
        range=[-1.1,1.1],  # sets the range of xaxis
        constrain="domain",  # meanwhile compresses the xaxis by decreasing its "domain"
    )
    fig.update_yaxes(
        scaleanchor = "x",
        scaleratio = 1
    )

    fig.update_traces(
        marker=dict(size=35, color=playlist_mood[2], line=dict(width=1, color='#160C28')),
        selector=dict(mode='markers'),
        hovertemplate="<br>".join([
            "Playlist",
            "Valência: %{x}",
            "Energia: %{y}"
        ])
    )

    fig2 = px.scatter(data_frame = outlier_data, x='valence_scaled', y='energy_scaled', labels={"x": "Valência", "y": "Energia"}, custom_data=['artist', 'track_title'])
    fig2.update_traces(
        hovertemplate="<br>".join([
            "Artista: %{customdata[0]}",
            "Faixa: %{customdata[1]}",
            "Valência: %{x}",
            "Energia: %{y}"
        ])
    )
    fig2.update_traces(
        marker=dict(size=10, color='black', line=dict(width=0, color='#160C28')),
        selector=dict(mode='markers')
    )
    
    fig.add_traces(
        list(fig2.select_traces())
    )

    fig3 = px.scatter(data_frame = inlier_data, x='valence_scaled', y='energy_scaled', labels={"x": "Valência", "y": "Energia"}, custom_data=['artist', 'track_title'])
    fig3.update_traces(
        hovertemplate="<br>".join([
            "Artista: %{customdata[0]}",
            "Faixa: %{customdata[1]}",
            "Valência: %{x}",
            "Energia: %{y}"
        ])
    )
    fig3.update_traces(
        marker=dict(size=10, color='black', line=dict(width=0, color='#160C28'), symbol="hexagram"),
        selector=dict(mode='markers')
    )
    
    fig.add_traces(
        list(fig3.select_traces())
    )

    return fig