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
    allow_origins=["*"],  # You can restrict this to specific domains later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🎬 Movie schema
class Movie(BaseModel):
    title: str
    genre: str
    cast: str
    director: str
    tagline: str
    image_url: str  # 👈 You’ll need to populate this with actual image links manually or with a lookup

# 📁 Load and preprocess data once on startup
df = pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/Movies%20Recommendation.csv')
df = df.fillna('').infer_objects(copy=False)
df_features = df[['Movie_Genre','Movie_Keywords','Movie_Tagline','Movie_Cast','Movie_Director']]
x = df_features['Movie_Genre'] + ' ' + df_features['Movie_Keywords'] + ' ' + df_features['Movie_Tagline'] + ' ' + df_features['Movie_Cast'] + ' ' + df_features['Movie_Director']
vectorizer = TfidfVectorizer()
feature_matrix = vectorizer.fit_transform(x)
similarity_score = cosine_similarity(feature_matrix)

# 🎯 Recommendation Logic
def recommend_movies(movie_name: str) -> List[Movie]:
    titles = df['Movie_Title'].tolist()
    close_match = difflib.get_close_matches(movie_name, titles, n=1)
    
    if not close_match:
        return []

    match = close_match[0]
    movie_id = df[df.Movie_Title == match]['Movie_ID'].values[0]
    score_list = list(enumerate(similarity_score[movie_id]))
    sorted_movies = sorted(score_list, key=lambda x: x[1], reverse=True)
    
    recommendations = []
    for i, (index, _) in enumerate(sorted_movies[:10]):
        movie_data = df[df.Movie_ID == index].iloc[0]
        movie_obj = Movie(
            title=movie_data['Movie_Title'],
            genre=movie_data['Movie_Genre'],
            cast=movie_data['Movie_Cast'],
            director=movie_data['Movie_Director'],
            tagline=movie_data['Movie_Tagline'],
            image_url=f"https://dummyimage.com/200x300/000/fff&text={movie_data['Movie_Title'].replace(' ', '+')}"
        )
        recommendations.append(movie_obj)
    return recommendations

# 🔗 API Endpoint
@app.get("/recommend", response_model=List[Movie])
def get_recommendations(movie: str = Query(..., description="Your favorite movie title")):
    return recommend_movies(movie)
uvicorn.run(app, host="127.0.0.1", port=8080)

