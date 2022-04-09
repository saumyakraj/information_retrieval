from django.shortcuts import render, redirect
from django.contrib import messages
# from .models import Image
from .forms import ImageForm
from django.core.files.storage import FileSystemStorage
import shutil
from .imagesearch import ImageSearch
import os
import time
from IR_Project.settings import IMG_SEARCH
import pandas as pd

# img_search = None

def upload_image(request):
    # global img_search
    # if img_search == None:
    #     img_search = ImageSearch()
    for root, dirs, files in os.walk('media/'):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
    if request.method == 'POST':
        upload = request.FILES['upload']
        fss = FileSystemStorage()
        file = fss.save(upload.name, upload)
        start = time.time()
        data = IMG_SEARCH.search_images(os.path.join("media" , file), 'inceptionresnetv2', 15)
        df = pd.read_csv(os.path.join('Data', 'captions.txt'))
        lis = []
        for i in data:
            ct = df.loc[df['image'] == i[1]].iloc[0]['caption']
            lis.append([i[0], i[1], ct])

        context = {'lis':lis, 'search_time':time.time() - start}
        return render(request, 'result.html', context)
    return render(request, 'home.html')
