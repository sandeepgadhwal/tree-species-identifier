import os

from django.shortcuts import render

# Create your views here.

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import torch

from django.conf import settings

from PIL import Image
import requests
import numpy as np
import pandas as pd

mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]

def read_image(url):
    r = requests.get(url, stream=True)
    im = Image.open(r.raw)
    if max(im.size) > 256:
        im = im.resize((256, 256))
    # im.save('/tmp/im.jpg')
    arr = np.array(im, dtype=np.uint8)
    arr = arr.astype(np.float32) / 255.
    arr = ( arr - mean_nums )/std_nums
    arr = np.rollaxis(arr, 2, 0)[None]
    return arr

class TreeModel(APIView):
    """
    Return the predicted class of tree species for the supplied image
    """
    def __init__(self):
        super().__init__()
        self.load_model()

    def load_model(self):
        print(f"-- Loading Model from path: {settings.MODEL_PATH}")
        if settings.MODEL_INSTANCE is None:
            print(f"-- Reading Model from path: {settings.MODEL_PATH}")
            settings.MODEL_INSTANCE = torch.load(settings.MODEL_PATH, map_location=torch.device('cpu'))
            settings.MODEL_INSTANCE = settings.MODEL_INSTANCE.eval()
            # ex = torch.rand((1, 3, 256, 256), dtype=torch.float32)
            # settings.MODEL_INSTANCE = torch.jit.trace(settings.MODEL_INSTANCE, ex)
            settings.CLASSES_DF = pd.read_csv(settings.MODEL_PATH.parent / 'classes.csv')
            settings.MODEL_DF = pd.read_csv(settings.MODEL_PATH.parent / 'Tree.csv').fillna('')

    def get(self, request):
        url = request.query_params.get('url', None)
        if url is None:
            message = 'This API requires url of image to be supplied.'
            return Response({"error": message}, status.HTTP_400_BAD_REQUEST)
        #
        arr = read_image(url)
        tsr = torch.tensor(arr, dtype=torch.float32)
        print(tsr.shape)
        res = settings.MODEL_INSTANCE(tsr)
        print('argmax', res.argmax().item())
        amax = res.argmax().item()
        return Response({
            # 'confidences': res.tolist(),
            'classindex': amax,
            'classname': settings.CLASSES_DF.classes[amax],
            'species': settings.MODEL_DF.iloc[amax].to_dict()
        })

class SpeciesList(APIView):
    """
    Return full list of tree species.
    """
    def get(self, request):
        return Response(settings.MODEL_DF.to_dict(orient='records'))
