from django.http import JsonResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser

from admin.myGAN.models import AutoencodersGans



@api_view(['GET'])
@parser_classes([JSONParser])
def autoencodersGans_process(request):
    AutoencodersGans().autoencodersGans_process()
    return JsonResponse({'AtoencodersGans': 'Success'})


from django.shortcuts import render

# Create your views here.
