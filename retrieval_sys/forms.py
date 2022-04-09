from django.forms import ModelForm
from django import forms
# from .models import Image
from django import forms

MODEL_CHOICES = [
    ('1', 'VGG19'),
    ('2', 'VGG16'),
    ('3', 'resnet'),
    ]
  
class ImageForm(forms.Form):
    upload = forms.ImageField()
    model = forms.ChoiceField(choices = MODEL_CHOICES)