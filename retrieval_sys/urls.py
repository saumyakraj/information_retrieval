from django.contrib import admin
from django.urls import path, include
from retrieval_sys import views

urlpatterns = [
    path('', views.home, name = 'home'),
    path('upload_image', views.upload_image, name='upload_image'),
]