from django.urls import path
from . import views

urlpatterns = [
    path('', views.home),
    path('upload/', views.upload),
    path('summarize/', views.summarize),
    path('upload/summarize/', views.summarize),
]