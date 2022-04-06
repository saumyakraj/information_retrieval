from django.shortcuts import render, redirect
from django.contrib import messages
# from .models import Image
from .forms import ImageForm
from django.core.files.storage import FileSystemStorage
import shutil
# Create your views here.

def home(request):
    return render(request, 'home.html', {})

def upload_image(request):
    shutil.rmtree('media/')
    if request.method == 'POST':
        print("####")
        print(request.FILES)
        print("####")
        upload = request.FILES['upload']
        fss = FileSystemStorage()
        file = fss.save(upload.name, upload)
        file_url = fss.url(file)
        return render(request, 'home.html', {'file_url': file_url})
    return render(request, 'home.html')