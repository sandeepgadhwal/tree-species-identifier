from django.urls import path
from django.conf.urls import url, include


from rest_framework_swagger.views import get_swagger_view

from .import views

schema_view = get_swagger_view(title='Trees API')

urlpatterns = [
    url(r'^$', schema_view),
    path('tree', views.TreeModel.as_view(), name='TreeModel'),
    path('species', views.SpeciesList.as_view(), name='SpeciesList'),

]