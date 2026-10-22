from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Movie schema
class Movie(BaseModel):
    title: str
    genre: str
    cast: str
    director: str
    tagline: str
    image_url: str 

# Load and preprocess data once on startup
df = pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/Movies%20Recommendation.csv')
df = df.fillna('').infer_objects(copy=False)
df_features = df[['Movie_Genre','Movie_Keywords','Movie_Tagline','Movie_Cast','Movie_Director']]
x = df_features['Movie_Genre'] + ' ' + df_features['Movie_Keywords'] + ' ' + df_features['Movie_Tagline'] + ' ' + df_features['Movie_Cast'] + ' ' + df_features['Movie_Director']
vectorizer = TfidfVectorizer()
feature_matrix = vectorizer.fit_transform(x)
similarity_score = cosine_similarity(feature_matrix)

# Recommendation Logic
def row_to_movie(row) -> Movie:
    return Movie(
        title=row['Movie_Title'],
        genre=row['Movie_Genre'],
        cast=row['Movie_Cast'],
        director=row['Movie_Director'],
        tagline=row['Movie_Tagline'],
        image_url=f"https://dummyimage.com/200x300/000/fff&text={row['Movie_Title'].replace(' ', '+')}"
    )

def recommend_movies(movie_name: str) -> List[Movie]:
    query = movie_name.strip()
    titles = df['Movie_Title'].tolist()
    close_match = difflib.get_close_matches(query, titles, n=1, cutoff=0.7)

    # 1) Title match -> similar movies
    if close_match:
        idx = df.index[df['Movie_Title'] == close_match[0]][0]
        scores = sorted(enumerate(similarity_score[idx]), key=lambda x: x[1], reverse=True)
        return [row_to_movie(df.iloc[i]) for i, _ in scores[1:11]]

    # 2) Fallback: search by actor or director name
    q = query.lower()
    mask = (
        df['Movie_Cast'].str.lower().str.contains(q, regex=False) |
        df['Movie_Director'].str.lower().str.contains(q, regex=False)
    )
    return [row_to_movie(r) for _, r in df[mask].head(10).iterrows()]

# API Endpoint
@app.get("/recommend", response_model=List[Movie])
def get_recommendations(movie: str = Query(..., description="Your favorite movie title")):
    return recommend_movies(movie)
uvicorn.run(app, host="127.0.0.1", port=8080)


