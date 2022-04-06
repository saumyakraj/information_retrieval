from django.forms import ModelForm
from django import forms
# from .models import Image
from django import forms

class ImageForm(forms.Form):
    upload = forms.ImageField()
