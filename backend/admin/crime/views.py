from django.http import JsonResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser
from admin.common.models import ValueObject, Reader, Printer
from admin.crime.models import CrimeCctvModel
import matplotlib.pyplot as plt
@api_view(['GET'])
@parser_classes([JSONParser])
def create_police_position(request):
    CrimeCctvModel().create_police_position()
    return JsonResponse({'result': 'Create Police Position Success'})

def create_cctv_model(request):
    CrimeCctvModel().create_cctv_model()
    return JsonResponse({'result': 'Create CCTV Model Success'})

def create_population_model(request):
    CrimeCctvModel().create_population_model()
    return JsonResponse({'result': 'Create Population Model Success'})