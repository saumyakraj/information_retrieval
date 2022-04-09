from django.shortcuts import render, redirect
from django.contrib import messages
# from .models import Image
from .forms import ImageForm
from django.core.files.storage import FileSystemStorage
import shutil
# from .imagesearch import ImageSearch
import os
# from IR_Project.settings import IMG_SEARCH
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
        model = request.POST['model']
        # print(upload)
        # print(model)
        fss = FileSystemStorage()
        file = fss.save(upload.name, upload)
        file_url = fss.url(file)
        # data = IMG_SEARCH.search_images(os.path.join("media" , file))
        data = [(0.9,'1.png'), (0.8,'2.png'), (0.7,'4.png')]
        context = {'lis':data}
        return render(request, 'result.html', context)
    context = {}
    context['form'] = ImageForm()
    return render( request, "home.html", context)
