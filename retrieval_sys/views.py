from django.shortcuts import render, redirect
from django.contrib import messages
# from .models import Image
from .forms import ImageForm
from django.core.files.storage import FileSystemStorage
import shutil
# from .imagesearch import ImageSearch
import os
import time
from IR_Project.settings import IMG_SEARCH
import pandas as pd


def upload_image(request):

    if request.method == 'POST':
        upload = request.FILES['upload']
        model = request.POST['model']
        fss = FileSystemStorage()
        file = fss.save(upload.name, upload)

        start = time.time()
        data = IMG_SEARCH.search_images(os.path.join("media" , file), 'inceptionresnetv2', 15)
        df = pd.read_csv(os.path.join('Data', 'captions.txt'))
        lis = []
        for i in data:
            ct = df.loc[df['image'] == i[1]].iloc[0]['caption']
            lis.append([i[0], i[1], ct])
        shutil.move(os.path.join('media', file), os.path.join(r"C:\Users\abhis\Desktop\IR\information_retrieval\retrieval_sys\static", file))
        context = {'lis':lis, 'search_time':time.time() - start, 'query_image':file}
        return render(request, 'result.html', context)
    context = {}
    context['form'] = ImageForm()
    return render( request, "home.html", context)
