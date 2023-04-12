from urllib import response
import streamlit as st
import pickle
import pandas as pd
import requests

def fetchMoviePoster(movie_id):
    response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=5c3c9860a7f043c8cf9c6ebcd61b4e91&language=en-US'.format(movie_id))

    data = response.json()
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']

similarity = pickle.load(open('../similarity.pkl','rb'))

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    recommend_movies = []
    recommend_movie_posters = []
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommend_movies.append(movies.iloc[i[0]].title)
        # Fetch poster from API
        recommend_movie_posters.append(fetchMoviePoster(movie_id))
    return recommend_movies,recommend_movie_posters

movie_dict = pickle.load(open('../movie_dict.pkl','rb'))
movies = pd.DataFrame(movie_dict)
st.title('Move Recommender System')

selected_movie_name = st.selectbox('Search Here',movies['title'].values)

if st.button('Recommend'):
    names,posters = recommend(selected_movie_name)

    col1,col2,col3,col4,col5 = st.beta_columns(5)
    with col1:
        st.text(names[0])
        st.image(posters[0])
    with col2:
        st.text(names[1])
        st.image(posters[1])
    with col3:
        st.text(names[2])
        st.image(posters[2])
    with col4:
        st.text(names[3])
        st.image(posters[3])
    with col5:
        st.text(names[4])
        st.image(posters[4])
