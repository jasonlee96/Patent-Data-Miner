from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from PatentMiner.query import PatentMining
import json
from PatentMiner.model_trainer import KNNModel
from .preprocessing import tokenize



def index(request):
    return render(request, 'index.html')

def query_term(request):
    term = request.GET.get('query', None)
    model = PatentMining()
    result = model.get_query(term)
    data = {
        'result': result
    }
    return HttpResponse(json.dumps(data), content_type="application/json")

def train(request):
    model = KNNModel()
    model.train()
    data = {
        'result': "OK"
    }
    return JsonResponse(data)
