from django.contrib import admin
from django.urls import path, include
from IR_Project import views

urlpatterns = [
    path('', views.index, name = 'home')
]