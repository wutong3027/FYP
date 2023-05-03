import io
import fitz
import nltk
import json
import re
from django.http import HttpResponse
from django.shortcuts import render
from django.http import HttpResponseBadRequest, HttpResponseRedirect
from django.http import JsonResponse
from ArticleSummarization.classes.user import User
from ArticleSummarization.classes.article import Article
from ArticleSummarization.classes.naiveBayes import NaiveBayes
from ArticleSummarization.classes.neuralNetwork import NeuralNetwork
from ArticleSummarization.classes.decisionTree import DecisionTree
nltk.download('punkt')
nltk.download('stopwords')

user = User()
nb = NaiveBayes()
nn = NeuralNetwork()
dt = DecisionTree()
# Create your views here.

def home(request):
    return render(request, 'home.html', {'name': 'Django'})

def upload(request):
    if request.method == 'POST':
        text = request.POST.get('text', '')
        pdf_file = request.FILES.get('pdf_file', None)
        text = user.upload_file(pdf_file)
        article = Article(pdf_file.name,text)
        return JsonResponse({'text': text})
    
    else:
        return render(request, 'home.html')

# Generate summaries
def summarize(request):
    # Get text from POST request
    text = request.POST.get('text', '')
    mode = request.POST.get('mode', 'naive_bayes') # Default to naive bayes if mode is not provided
    summary = user.summarize(text , mode)
    return JsonResponse({'summary': summary})