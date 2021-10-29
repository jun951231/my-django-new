from django.http import JsonResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser
from admin.common.models import ValueObject, Reader, Printer
from admin.crawling.models import Crawling, NewsCrawling

import matplotlib.pyplot as plt


@api_view(['GET'])
@parser_classes([JSONParser])
def process(request):
    Crawling().process()
    return JsonResponse({'result': 'Create Crawling Success'})

@api_view(['GET'])
@parser_classes([JSONParser])
def process(request):
    NewsCrawling().get()
    return JsonResponse({'result': 'Create Crawling Success'})