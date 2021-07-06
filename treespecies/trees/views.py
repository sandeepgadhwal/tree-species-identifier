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


def read_image(url):
    r = requests.get(url, stream=True)
    arr = np.array(Image.open(r.raw), dtype=np.uint8)
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
            ex = torch.rand((1, 3, 256, 256), dtype=torch.float32)
            settings.MODEL_INSTANCE = torch.jit.trace(settings.MODEL_INSTANCE, ex)
            settings.MODEL_DF = pd.read_csv(settings.MODEL_PATH.parent / 'Tree.csv')

    def get(self, request):
        url = request.query_params.get('url', None)
        if url is None:
            message = 'This API requires url of image to be supplied.'
            return Response({"error": message}, status.HTTP_400_BAD_REQUEST)
        #
        arr = read_image(url)
        tsr = torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1)[None]
        res = settings.MODEL_INSTANCE(tsr)
        return Response({
            'confidences': res.tolist(),
            'argmax': settings.MODEL_DF.iloc[res.argmax().item()].to_dict()
        })