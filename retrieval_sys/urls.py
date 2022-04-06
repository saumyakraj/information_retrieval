from django.contrib import admin
from django.urls import path, include
from retrieval_sys import views

urlpatterns = [
    path('', views.upload_image, name='upload_image'),
]