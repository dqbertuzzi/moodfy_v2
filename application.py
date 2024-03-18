import os
from sqlalchemy import create_engine
from sqlalchemy import text
import psycopg2
import pandas as pd
import random

# Dash and related imports
from dash import Dash, html, dash_table, dcc, callback, Output, Input, State, no_update
import dash_bootstrap_components as dbc

# Other specific library imports
from dotenv import load_dotenv
from utils import bertify

load_dotenv()

app = Dash(__name__,
          meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0, maximum-scale=1.2, minimum-scale=0.5,'}], external_stylesheets=[dbc.themes.MINTY])

server = app.server
app.title = "Moodfy"
app.config.suppress_callback_exceptions = True

def display_artistInfo(info):
    elements = [html.H2("Estas são as faixas que não combinam com a emoção da playlist:",
            style={'color':'#191414',"font-size":"2.8vh"})]
    for e in range(len(info['artists'])):
        elements.append(html.Br())
        elements.append(html.Img(src=info['img_srcs'][e], style={'height':'35px', 'width':'35px', 'vertical-align': 'middle', 'display':'inline-block'}))
        elements.append(html.Div(html.P(info['artists'][e] + " - " + info['track_titles'][e]),style={'display': 'inline-block', 'padding-left': '.5vw'}))
        
    return elements

def display_inliers(info):
    elements = [html.H2("Estas são as faixas recomendadas baseadas na emoção e estilo musical da playlist:",
            style={'color':'#191414',"font-size":"2.8vh"})]
    for e in range(len(info['artists'])):
        elements.append(html.Br())
        elements.append(html.Img(src=info['img_srcs'][e], style={'height':'35px', 'width':'35px', 'vertical-align': 'middle', 'display':'inline-block'}))
        elements.append(html.Div(html.P(info['artists'][e] + " - " + info['track_titles'][e]),style={'display': 'inline-block', 'padding-left': '.5vw'}))
        
    return elements

tab_style = {
    'font-size': '2.7vh'
}

tab_selected_style = {
    'backgroundColor': '#191414',
    'color': 'white',
}

app.layout = html.Div([
    dcc.Store(id='dataset', storage_type='memory'),
    dcc.Store(id='dataset_db', storage_type='memory'),
    dcc.Store(id='playlistMood', storage_type='memory'),
    html.Div(
    html.H1("Moodfy - Spotify Mood Check",
            style={'color':'#1db954', "font-size":"4vh"}), style={'display': 'inline-block'}
            ),
    html.P(["Criado por ", html.A("Daniel", href="https://github.com/dqbertuzzi")],
           style={"font-size":"2vh"}),
    html.Div(
    html.Hr(style={'border':'none', 'height':'.3vh', 'background-color':'#160C28', 'display': 'block'})),
    html.H2("Sintonize sua emoção: descubra o sentimento por trás da sua playlist!",
            style={'color':'#191414',"font-size":"2.8vh", "margin-bottom":"2vh", 'margin-left':'1.5vw','margin-right':'1.5vw','line-break': 'strict'}),
    dbc.Tabs([
        dbc.Tab([
            html.Br(),
            dcc.Input(
                id="input_playlistUrl",
                type='text',
                placeholder="Link da playlist", style={'width':'60vw','display': 'inline-block', 'vertical-align': 'middle'}
                ),
            html.Div([
                dbc.Button("Enviar", color="dark", className="me-1", id='submit-button')],
                style={'margin-left':'1vw',
                       'display': 'inline-block',
                       'vertical-align': 'middle'}
                       ),
            dbc.Modal([dbc.ModalHeader(dbc.ModalTitle("Servidor sobrecarregado!")),
                       dbc.ModalBody(["Volte em alguns minutos e tente de novo."])],
            id='modal1', is_open=False),
            dbc.Modal([dbc.ModalHeader(dbc.ModalTitle("Link inválido!")),
                       dbc.ModalBody([html.Ul([html.Li("O link deve estar no formato https://open.spotify.com/playlist/XXXXXXXXXXXXXXXXXXXXXX?si=XXXXXXXXXXXXXXXX"),
                                               html.Li("A playlist deve estar configurada como pública.")])])],
                       id='modal2', is_open=False, size="lg",),
            html.Div([
                dbc.Spinner(html.Div(id="loading-output-1",
                                     style={'display': 'inline-block', 'padding-left': '50px'}))],
                                     style={'display': 'inline-block','vertical-align': 'middle'}),
            html.Br(),
            html.Div([
                html.Br(),
                html.P("O gráfico exibe onde sua playlist se situa emocionalmente (círculo maior colorido) e as faixas musicais que não combinam (círculos) e recomendadas (estrelas):",
                        id='mood-h2', style={'display':'none', 'color':'#191414',"font-size":"3vh", 'margin-left':'1.5vw','margin-right':'1.5vw','line-break': 'strict'}),
                dcc.Graph(id='graph-figure', style={'display': 'none'})],
                id="mood-div"),
            html.Div(id='output-outliers', style={
                                                     'vertical-align': 'middle', "margin-bottom":"2vh", 'margin-left':'1.5vw'}
                                                     ),
            html.Div(id='output-inliers', style={
                                                 'display': 'inline-block','vertical-align': 'middle', "margin-bottom":"2vh", 'margin-left':'1.5vw'}
                                                 )],label='Análise', tab_style=tab_style),
        dbc.Tab([
            html.Div([
            html.Br(),
            html.H2("Sobre o Modelo Circumplexo de Emoção", style={'color':'#191414',"font-size":"3vh"}),
            html.P("Na década de 1980, Russell (1980) apresentou uma contribuição teórica para a compreensão do afeto, por meio da caracterização de duas dimensões: valência e ativação. As combinações dessas duas dimensões, em diferentes graus, teriam como resultado as experiências afetivas. O modelo teórico resultante, denominado de circumplexo de Russell (1980), teve continuidade em estudos posteriores (Carroll, Yik, Russell, & Barrett, 1999; Russell & Barrett, 1999; Russell, 2003; Yik, Russell, & Steiger, 2011). Estudos como os de Carroll et al. (1999) e Yik et al. (2011) trouxeram avanços com relação a como modelar as variáveis de afeto em um circumplexo e como sentimentos são entendidos por esse modelo; pesquisas como a de Russell e Barrett (1999) visavam indicar novas perspectivas sobre como medir afeto e discutir esse construto teoricamente. Ao longo das décadas houve mudanças importantes na teoria do afeto, especialmente ao se reconhecer que os estudos sobre humor pertenciam ao campo conceitual do afeto.", style={'color':'#191414',"font-size":"2.8vh",'text-align':'justify'}),
            html.P("O afeto, para Russell (1980), é compreendido por meio do circumplexo (Figura 1). Suas dimensões são bipolares e ortogonais, sendo nomeadas de valência (prazer ou desprazer) e ativação percebida (alta ou baixa). O circumplexo é uma estrutura ordenada em que todos os testes apresentam um mesmo nível de complexidade e diferem em termos do tipo de habilidade que eles medem. Quando um construto pode ser representado por um circumplexo, sua matriz de correlações apresenta um padrão de correlações fortes perto da diagonal e, conforme as correlações se afastam da diagonal, elas ficam mais fracas, até que voltam a ficar fortes. Esse padrão de correlações repete-se em toda a matriz, e, por isso, pontos próximos no circumplexo são correlacionados fortemente (Guttman, 1954).", style={'color':'#191414',"font-size":"2.8vh",'text-align':'justify'}),
            html.P([html.Img(src='assets/n2a05f1.jpg', alt='circumplexo', style={'height':'300px', 'width':'300px'})], style={"margin-bottom":"2vh", 'margin-left':'1.5vw','margin-right':'1.5vw', 'text-align':'center'}),
            html.P("A dimensão valência está relacionada à codificação do ambiente como prazeroso ou desprazeroso. Para o estímulo em um determinado momento, o sujeito pode atribuir um significado: bom ou ruim; útil ou prejudicial; recompensador ou ameaçador (Barrett, 2006). A ativação, por sua vez, é a dimensão da experiência que corresponde à mobilização ou energia dispensada; ou seja, é representada por um continuum, desde a baixa ativação, representada por sono, até a ativação alta, representada pela excitação (Russell & Barrett, 1999).", style={'color':'#191414',"font-size":"2.8vh",'text-align':'justify'}),
            html.P("Estados afetivos que são próximos no circumplexo representam uma combinação similar de valência e ativação percebida; já estados afetivos posicionados diametricamente longe um do outro diferem em termos de valência e ativação (Russell, 1980). Assim, as quatro variáveis alocadas diagonalmente não são dimensões, mas ajudam a definir os quadrantes no espaço do circumplexo (Russell, 1980).", style={'color':'#191414',"font-size":"2.8vh",'text-align':'justify'}),
            html.P(html.A("Crispim, Ana Carla, Cruz, Roberto Moraes, Oliveira, Cassandra Melo, & Archer, Aline Battisti. (2017). O afeto sob a perspectiva do circumplexo: evidências de validade de construto. Avaliação Psicológica, 16(2), 145-152. https://dx.doi.org/10.15689/AP.2017.1602.04", href='http://dx.doi.org/10.15689/AP.2017.1602.04', style={"font-size":"2vh",'text-align':'justify'}),
                   style={'padding-left':'15vw', 'text-align':'justify'}), 
        ], style={"margin-bottom":"2vh", 'margin-left':'1.5vw','margin-right':'1.5vw', 'line-break': 'strict'})
        ], label="Saiba mais", tab_style=tab_style)])
    ]
    ,
    style={'margin-left': '2.5vw',
           'margin-top': '2.5vw',
           'margin-bottom': '2.5vw',
           'margin-right':'2.5vw'}
)

@callback(
        [Output("loading-output-1", "children", allow_duplicate=True),
        Output("modal1", "is_open"),
        Output("modal2", "is_open"),
        Output("dataset", 'data'),
        Output("dataset_db", 'data'),
        Output("playlistMood", 'data')],
        Input('submit-button', 'n_clicks'),
        State('input_playlistUrl', 'value'),
        prevent_initial_call=True
        )
def update_output(clicks, input_value):
    if clicks is not None:
        client_id = os.getenv("CLIENT_ID")
        client_secret = os.getenv("CLIENT_SECRET")
        token = bertify.get_token(client_id, client_secret)
        headers = bertify.get_auth_header(token)
        
        url = f'{input_value}'
        
        final_dataset = bertify.get_final_dataset(url, headers)

        if isinstance(final_dataset, str):
            if final_dataset=='429':
                return None, True, False, no_update, no_update, no_update
            else:
                return None, False, True, no_update, no_update, no_update
        
        else:
            # Load new data to database
            table_name = os.getenv("TABLE_NAME")
            db_host = os.getenv("DB_HOST")
            db_user = os.getenv("DB_USER")
            db_passwd = os.getenv("DB_PASS")
            engine = create_engine(f"postgresql://{db_user}:{db_passwd}@{db_host}/")

            cols = ['artist', 'artist_id', 'album_image', 'track_title', 'track_href', 'danceability',
                    'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
                    'instrumentalness', 'liveness', 'valence', 'tempo', 'type', 'id', 'uri']

            with engine.begin() as conn:
                final_dataset[cols].to_sql(name='temp', con=conn, index=False)
                query = '''
                    SELECT * FROM temp
                    EXCEPT
                    SELECT * FROM music_data
                '''

                new_entries = pd.read_sql(query, con=conn)

                new_entries.to_sql(name=table_name, con=conn,  index=False)

                conn.execute(text("DROP TABLE temp;"))
                results = conn.execute(text("SELECT * FROM music_data"))

                df_music_data = pd.DataFrame(results.fetchall(), columns=cols)
            engine.dispose()

            # Calculate metrics
            final_dataset, scaler = bertify.scale_columns(final_dataset)
            final_dataset, playlist_mood, median = bertify.calculate_metrics(final_dataset)

            # Calculate Playlist tf-idf
            final_dataset_genres = bertify.get_genres(final_dataset['artist_id'].unique(), headers=headers)
            final_dataset_genres = final_dataset_genres['genres'].apply(lambda x: ['_'.join(word.split()) for word in x])
            playlist_genres_ = [i for i in final_dataset_genres]
            
            playlist_dff = pd.DataFrame({'artist_id':'playlist', 'genres':[[genre for sublist in playlist_genres_ for genre in sublist]]})

            # Calculate database tf-idf
            ## Getting random artists_ids, with a max of 100 entries (spotify API limits to 50 ids)
            artists = df_music_data['artist_id'].unique()
            random.shuffle(artists)
            artists_to_keep = artists[:200]

            df_music_data = df_music_data[df_music_data['artist_id'].isin(artists_to_keep)]
            music_data_genres = bertify.get_genres(df_music_data['artist_id'].unique(), headers=headers)
            music_data_genres['genres'] = music_data_genres['genres'].apply(lambda x: ['_'.join(word.split()) for word in x])
            music_data_genres = pd.concat([music_data_genres, playlist_dff], axis=0, ignore_index=True)

            genre_tfidf = bertify.get_tdidf(dataset=music_data_genres, genre_column='genres')
            music_data_genres = music_data_genres.join(genre_tfidf)

            playlist_dff = music_data_genres[music_data_genres['artist_id']=='playlist']

            df_music_data = df_music_data.merge(music_data_genres, on='artist_id')
            import numpy as np
            # Wrap it all
            df_music_data = df_music_data[~df_music_data['track_href'].isin(final_dataset['track_href'])]
            scaled_music_data = np.round(scaler.transform(df_music_data[['valence', 'energy']]),2)
            scaled_music_data = pd.DataFrame(scaled_music_data, columns=[f'{col}_scaled' for col in ['valence', 'energy']])
            df_music_data = pd.concat([df_music_data, scaled_music_data], axis=1)

            #df_music_data = (df_music_data.merge(final_dataset,
            #                                     on = ['artist', 'artist_id', 'album_image', 'track_title', 'track_href', 'danceability',
            #                                           'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
            #                                           'instrumentalness', 'liveness', 'valence', 'tempo', 'type', 'id', 'uri', 'valence_scaled', 'energy_scaled'],
            #                                     how = 'left',
            #                                     indicator = True).query('_merge == "left_only"').drop(columns='_merge')
            #                                     )
            
            df_music_data = bertify.db_metrics(df_music_data, median, playlist_dff)

            return None, False, False, final_dataset.set_index('track_href').to_dict('records'), df_music_data.set_index('track_href').to_dict('records'), playlist_mood
        
@callback(
    [Output("loading-output-1", "children"),
     Output('graph-figure', 'figure'),
     Output('graph-figure', 'style'),
     Output('output-outliers', 'children'),
     Output('output-inliers', 'children'),
     Output('mood-h2', 'style')],
     Input('dataset', 'data'),
     Input('dataset_db', 'data'),
     Input('playlistMood', 'data'),
     prevent_initial_call=True
     )
def display_results(dataset, dataset_db, playlistMood):
    dff = pd.DataFrame(dataset)
    dff_db = pd.DataFrame(dataset_db)

    outlier_data = dff[dff['outlier'] == 1]
    outlier_artists = {'artists': list(outlier_data['artist']),
               'track_titles': list(outlier_data['track_title']),
               'img_srcs': list(outlier_data['album_image'])}
    
    inlier_data = dff_db[dff_db['inlier'] == 1].sort_values(by='score', ascending=False).drop_duplicates('artist_id').head(len(outlier_data))
    inlier_artists = {'artists': list(inlier_data['artist']),
               'track_titles': list(inlier_data['track_title']),
               'img_srcs': list(inlier_data['album_image'])}
    
    return None, bertify.create_mood_scatter_plot(playlistMood, outlier_data, inlier_data), {'display': 'block'}, display_artistInfo(outlier_artists), display_inliers(inlier_artists), {'display': 'block'}
    
if __name__ == "__main__":
    app.run(debug=True)
