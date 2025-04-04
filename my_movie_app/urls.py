from django.urls import path
from .views import recommend_movies  # Import your view function

urlpatterns = [
    path('recommend/', recommend_movies, name='recommend_movies'),
]
