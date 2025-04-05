import pandas as pd
import requests
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# TMDB API Key
TMDB_API_KEY = "23a06b4557ab8c23deb7834bd0129046"


# Load cleaned dataset
movies = pd.read_csv("cleaned_movies.csv")
movies['title'] = movies['title'].str.lower() 
movies['tags'] = movies['tags'].fillna('')

# TF-IDF and Cosine Similarity
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies['tags'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def fetch_poster(title):
    try:
        title = title.lower()
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
        response = requests.get(url).json()
        print(f"TMDB response for '{title}':", response)  # Debug log

        # Check if 'results' exists and has at least one item
        if response.get("results") and len(response["results"]) > 0:
            poster_path = response["results"][0].get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
        return None
    except Exception as e:
        print("Error fetching poster:", e)
        return None


@csrf_exempt
def recommend_movies(request):
    if request.method == "GET":
        movie_name = request.GET.get("title")

        if movie_name not in movies['title'].values:
            return JsonResponse({"error": "Movie not found"}, status=404)

        # Get index of the searched movie
        idx = movies[movies['title'] == movie_name].index[0]

        # Get similarity scores
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Exclude the searched movie itself and take top 5 similar
        recommended_movie_indices = [i[0] for i in sim_scores if i[0] != idx][:5]

        # Prepare response starting with the searched movie
        searched_movie_poster = fetch_poster(movie_name)
        recommendations = [{
            "title": movie_name,
            "poster": searched_movie_poster
        }]

        # Add top 5 similar movies
        for index in recommended_movie_indices:
            title = movies.iloc[index]['title']
            poster_url = fetch_poster(title)
            recommendations.append({
                "title": title,
                "poster": poster_url
            })

        return JsonResponse(recommendations, safe=False)