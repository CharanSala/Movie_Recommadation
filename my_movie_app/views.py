import pandas as pd
import requests
import difflib  
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
        print(title)
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
        response = requests.get(url).json()
        

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
        movie_name = movie_name.lower() if movie_name else ""

        # Find close match to the movie title in the dataset
        all_titles = movies['title'].str.lower().tolist()
        close_matches = difflib.get_close_matches(movie_name, all_titles, n=1, cutoff=0.6)

        if not close_matches:
            return JsonResponse({"error": "Movie not found"}, status=404)

        matched_movie_name = close_matches[0]

        # Get index of the matched movie
        idx = movies[movies['title'].str.lower() == matched_movie_name].index[0]

        # Get similarity scores
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Exclude the matched movie itself and take top 5 similar
        recommended_movie_indices = [i[0] for i in sim_scores if i[0] != idx][:5]

        # Prepare response
        searched_movie_poster = fetch_poster(matched_movie_name)
        recommendations = [{
            "title": matched_movie_name,
            "poster": searched_movie_poster
        }]

        for index in recommended_movie_indices:
            title = movies.iloc[index]['title']
            poster_url = fetch_poster(title)
            recommendations.append({
                "title": title,
                "poster": poster_url
            })

        return JsonResponse(recommendations, safe=False)
