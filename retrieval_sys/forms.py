from django.forms import ModelForm
from django import forms
# from .models import Image
from django import forms

MODEL_CHOICES = [
    ('1', 'vgg19'),
    ('2', 'vgg16'),
    ('3', 'resnet'),
    ('4', 'inceptionresnetv2'),
    ]
  
class ImageForm(forms.Form):
    upload = forms.ImageField()
    model = forms.ChoiceField(choices = MODEL_CHOICES)
    num_of_images = forms.IntegerField(max_value=100, min_value=1, required=True)