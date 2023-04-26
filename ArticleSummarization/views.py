import io
import fitz
import nltk
import json
import re
nltk.download('punkt')
from django.http import HttpResponse
from django.shortcuts import render
from django.http import HttpResponseBadRequest
from django.http import JsonResponse
from sklearn.naive_bayes import MultinomialNB
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import heapq
from sklearn.feature_extraction.text import CountVectorizer

from transformers import AutoModelForSeq2SeqLM, BartTokenizer

import string
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction.text import CountVectorizer

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Create your views here.

def home(request):
    return render(request, 'home.html', {'name': 'Django'})

filepath = ""
def upload(request):
    if request.method == 'POST':
        text = request.POST.get('text', '')
        pdf_file = request.FILES.get('pdf_file', None)
        print(pdf_file.name)
        if pdf_file is not None:
            with io.BytesIO(pdf_file.read()) as pdf_buffer:
                pdf_doc = fitz.open(stream=pdf_buffer.read(), filetype="pdf")
                for page_num in range(pdf_doc.page_count):
                    page = pdf_doc.load_page(page_num)
                    text += page.get_text()
        return render(request, 'home.html', {'text': text})
    else:
        return render(request, 'home.html')

# Generate summaries
def summarize(request):
    # Get text from POST request
    text = request.POST.get('text', '')
    mode = request.POST.get('mode', 'naive_bayes') # Default to naive bayes if mode is not provided

    # sentences = nltk.sent_tokenize(text)

    # print(sentences)

    # Define section keywords
    section_keywords = {
        'introduction': ['introduction', 'background', 'motivation'],
        'related work': ['related work', 'previous work', 'literature review'],
        'methodology': ['methodology', 'method', 'approach'],
        'results': ['results', 'findings', 'experiment'],
        'discussion': ['discussion', 'conclusion', 'implications'],
        'acknowledgements': ['acknowledgments', 'acknowledgements'],
    }
    #     # Split text into sections based on section headers
    sections = {}
    current_section = None
    for line in text.splitlines():
        # Check if line matches any of the section headers
        if any(header.lower() in line.lower() for header in section_keywords) and len(line.split()) <= 3:
            # If so, start a new section
            print(len(line.split()))
            current_section = line.strip()
            sections[current_section] = ""
        # Otherwise, add the line to the current section (if one exists)
        elif current_section is not None:
            sections[current_section] += line.strip() + " "
        # If the line doesn't match any section header and a section is currently open,
        # add the line to the current section. Otherwise, create a new "Miscellaneous" section.
        else:
            if "Overall" not in sections:
                sections["Overall"] = ""
            sections["Overall"] += line.strip() + " "

    print("-------------------------")
    print("Collected section headers:", list(sections.keys()))
    #print(sections)
    for text in sections:
        # print(sections.keys())
        print(text)
        print(sections[text])
        print()


    # # Identify sections
    # section_labels = []
    # for sentence in sentences:
    #     for section, keywords in section_keywords.items():
    #         if any(re.search(keyword, sentence, re.IGNORECASE) for keyword in keywords):
    #             section_labels.append(section)
    #             break
    #     else:
    #         section_labels.append('miscellaneous')

    # # Group sentences by section
    # section_sentences = {}
    # for sentence, section in zip(sentences, section_labels):
    #     section_sentences.setdefault(section, []).append(sentence)

    # # Generate summary for each section
    # section_summaries = {}
    # for section, sentences in section_sentences.items():
    #     summary = NB_generate_summary(' '.join(sentences))
    #     section_summaries[section] = summary

    section_summaries = {}
    for text in sections:
        section_summaries[text] = NN_generate_summary(sections[text])
          
    # Join section summaries with line breaks
    summary = '\n'.join([f"{section.upper()}\n{section_summaries[section]}\n" for section in section_summaries])
    print(summary)
    # summary = NB_generate_summary(text)
    # summary = ""
    return JsonResponse({'summary': summary})

def NB_generate_summary(text):
    num_sentences = 3
    if not isinstance(text, str):
        raise TypeError('Input must be a string')
    if not isinstance(num_sentences, int) or num_sentences < 1:
        raise ValueError('Number of sentences must be a positive integer')
    
    # tokenizing the text by sentence
    sentences = sent_tokenize(text)
    
    # removing stop words
    # stop_words = set(stopwords.words('english'))
    # words = word_tokenize(text)
    # words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
    words = re.sub('[^a-zA-Z]', ' ', text)
    words = words.lower()
    words = words.split()
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words if not word in set(stopwords.words('english'))]
    words = ' '.join(words)
    
    # calculating the word frequency distribution
    freq_dist = FreqDist(words)
    
    # feature extraction
    cv = CountVectorizer(stop_words='english')
    sentence_features = cv.fit_transform(sentences)
    
    # creating labels for training data
    summary_sentences_idx = heapq.nlargest(num_sentences, range(len(sentences)), key=lambda i: sentence_features[i].sum())
    labels = ['summary' if i in summary_sentences_idx else 'non-summary' for i in range(len(sentences))]
    
    # training the classifier
    clf = MultinomialNB()
    clf.fit(sentence_features, labels)
    
    # selecting the top 'num_sentences' sentences based on the classifier scores
    sentence_scores = clf.predict_proba(sentence_features)[:, 0]
    summary_sentences_idx = heapq.nlargest(num_sentences, range(len(sentences)), key=lambda i: sentence_scores[i])
    summary_sentences = [sentences[idx] for idx in sorted(summary_sentences_idx)]
    
    # combining the selected sentences to generate the summary
    summary = ' '.join(summary_sentences)
    return summary

def NN_generate_summary(text):
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large-cnn')

    # text = "Signals are only permitted to move in one direction via feed-forward neural networks, which is from the input to the output. A network that feeds data from the input layer to the output layer is known as a feedforward neural network (Dematos, Boyd, Kermanshahi, Kohzadi, & Kaastra, 1996). There is no feedback, hence the fact that the output of one layer does not have an effect on that same layer is not a problem. In most cases, feed-forward networks are straightforward configurations that link inputs and outputs directly. From the approach above, seven characteristics were derived from the sentences that were provided as input. After the network has learned the characteristics that most accurately describe the summary phrase, a process known as feature fusing is carried out by eliminating and merging certain features. The most important step is called the feature fusion phase, and it's during this phase that the relationships between the features are figured out in two phases. 1) removing elements that are used seldom 2) reducing common characteristics into fewer groups before evaluating the sentences to determine which ones are the most significant summary sentences. After that, the pruned network model is used to identify the summary sentences.Furthermore, Svore et al. (2007) at the Microsoft Research Department created a system for the summarization of a single document that is known as NetSum. A neural network model was used in the development of the NetSum system, which was designed to generate summaries automatically (Yogan et al., 2016). The NetSum system is a single-document summarizer that makes use of neural networks to improve sentence features. The network model is trained with the help of the training set, which consists of articles obtained from CNN.com suggest by the author. Using a neural network that contains a gradient descent approach for the training, the trained model was able to infer the correct ranking of sentences inside a test document. A sentence containing the keyword search entered by the user has a better probability of being included in the generated output summary (Svore et al., 2007). It employs machine learning methods by labelling a train set such that the labels determine the most effective sentences. It then takes up to three phrases from the text that are the most relevant to its highlights. After that, the trained model is applied to the task of ranking newly created sentences. The RankNet algorithm (Burges, Shaked, Renshaw, Lazier, Deeds, Hamilton & Hullender, 2005) is used in the process of sentence ranking carried out by the NetSum system.Decision tree learning is one of the most used and useful techniques for inductive approach (Mitchell, 1997). It is a technique for approximating discrete-valued functions that is able to learn disjunctive formulations and is robust to noisy input. In this technique, the learnt function is represented by a decision tree, and the technique is called decision tree learning. In order to make learned trees more comprehensible to humans, they may also be represented as sets of if-then rules. Breaking a data set down into smaller subsets while concurrently creating the corresponding decision tree is the process that decision trees use to construct a classification and regression tree model in the form of a tree structure. The decision tree is a hierarchical structure with one root node, and it is breaking its parent-child-related branches. The root node defining a testing condition which outcome corresponds to a branch leading to an internal node. Classifications are assigned to the terminal nodes of the tree in figure 2.3, also known as the leaf nodes. The Decision Node and the Leaf Node are the two different types of nodes that may be found in a decision tree. It is a classifier in the form of a tree, with internal nodes representing the characteristics of a dataset, branches representing the decision rules, and each leaf node representing the conclusion of the classification. Decision nodes are used to make decision and have numerous branches, whereas Leaf nodes represent the results of these decisions and do not contain any more branches."

    if text is not None:
        input_ids = tokenizer(text, truncation=True, max_length=1024, padding='max_length', return_tensors='pt')
        summary_ids = model.generate(input_ids['input_ids'], num_beams=10, max_length=512)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

def DT_generate_summary(text):
    sentences = nltk.sent_tokenize(text)

    # Preprocess the text
    punctuation = string.punctuation.replace(".", "")  # Remove periods from punctuation
    processed_sentences = []
    for sentence in sentences:
        nopunc = "".join([char for char in sentence if char not in punctuation])
        processed_sentences.append(nopunc.lower())

    # Create BoW feature vectors
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(processed_sentences)

    # Train a decision tree to predict sentence importance
    y = [len(sentence) for sentence in sentences]  # Use sentence length as target variable
    dt = DecisionTreeRegressor()
    dt.fit(X, y)

    # Use the decision tree to predict sentence importance scores
    scores = dt.predict(X)

    # Select the top-scoring sentences as the summary
    num_sentences = 3  # Set the desired number of summary sentences
    summary_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:num_sentences]
    summary_sentences = [sentences[i] for i in summary_indices]
    summary = " ".join(summary_sentences)

    return summary















# new_text = "Some text to be summarized"
# new_tokenized_text = new_text.split()
# new_X = vectorizer.transform([new_tokenized_text])
# predicted_summary = clf.predict(new_X)[0]

# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.feature_extraction.text import TfidfVectorizer

# def summarize(request):
#     # Get text from POST request
#     text = request.POST.get('text', '')
    
#     # Load pre-trained model and tokenizer
#     model_name = "facebook/bart-large-cnn"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

#     # Tokenize input and encode summary
#     inputs = tokenizer(text, truncation=True, max_length=512, padding='max_length', return_tensors='pt')
#     summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=100, early_stopping=True)
#     summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

#     # Load data and preprocess
#     data = [text, summary]
#     vectorizer = TfidfVectorizer(stop_words='english', strip_accents='unicode', max_features=512)
#     X = vectorizer.fit_transform(data).toarray()

#     # Initialize model and predict
#     model = DecisionTreeClassifier()
#     model.fit(X, [0, 1])
#     prediction = model.predict(X[-1].reshape(1, -1))

#     # Return summary if prediction is positive, else return original text
#     if prediction == 1:
#         return JsonResponse({'summary': summary})
#     else:
#         return JsonResponse({'summary': text})


# def summarize(request):
#     # Get text from POST request
#     text = request.POST.get('text', '')
    
#     # Load pre-trained model and tokenizer
#     model_name = "facebook/bart-large-cnn"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

#     # Tokenize input
#     inputs = tokenizer(text, truncation=True, max_length=512, padding='max_length', return_tensors='pt')

#     # Generate summary
#     summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=100, early_stopping=True)
#     summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#     print("--------------------------")
#     print(summary)
#     return JsonResponse({'summary': summary})



# section_headers = ["INTRODUCTION", "RELATED WORK", "RELATED WORKS", "METHODOLOGY", "RESULTS", "CONCLUSION", "DISCUSSION", "FUTURE WORK", "ACKNOWLEDGEMENTS", "REFERENCES"]


# def summarize(request):
#     # Get text and mode from POST request
#     text = request.POST.get('text', '')
#     mode = request.POST.get('mode', 'naive_bayes') # Default to naive bayes if mode is not provided
#     # Split text into sections based on section headers
#     sections = {}
#     current_section = None
#     for line in text.splitlines():
#         # Check if line matches any of the section headers
#         if any(header.upper() in line.upper() for header in section_headers):
#             # If so, start a new section
#             current_section = line.strip()
#             sections[current_section] = ""
#         # Otherwise, add the line to the current section (if one exists)
#         elif current_section is not None:
#             sections[current_section] += line.strip() + " "

#     print("-------------------------")
#     print(sections)


#     # Process each section using the appropriate classifier pipeline
#     summaries = {}
#     for header, section_text in sections.items():
#         # print("-------------------------")
#         # print(header)
#         if header not in section_headers:
#             section_headers.append(header)

#     pipelines = {}
#     for header in section_headers:
#         pipelines[header] = Pipeline([
#             ('tfidf', TfidfVectorizer(stop_words='english')),
#             ('clf', MultinomialNB())
#         ])
#         # print("-------------------------")
#         # print(header)
#     for header, section_text in sections.items():
#         # Skip the abstract and references sections
#         if header.upper() == "ABSTRACT" or header.upper() == "REFERENCES" or header.upper() not in section_headers:
#             continue
        
#         section_headers.append(header.upper())
#         # Preprocess text and fit vectorizer to data
#         sentences = nltk.sent_tokenize(section_text)
#         pipelines[header].named_steps['tfidf'].fit(sentences)
#         X = pipelines[header].named_steps['tfidf'].transform(sentences)

#         # Select classifier based on mode and fit classifier to data
#         if mode == 'naive_bayes':
#             pipelines[header].named_steps['clf'] = MultinomialNB()
#             pipelines[header].named_steps['clf'].fit(X, range(len(sentences)))
#         elif mode == 'neural_network':
#             pipelines[header].named_steps['clf'] = MLPClassifier()
#             pipelines[header].named_steps['clf'].fit(X, range(len(sentences)))
#         elif mode == 'decision_tree':
#             pipelines[header].named_steps['clf'] = DecisionTreeClassifier()
#             pipelines[header].named_steps['clf'].fit(X, range(len(sentences)))
#         else:
#             # Default to naive bayes if mode is not recognized
#             pipelines[header].named_steps['clf'] = MultinomialNB()
#             pipelines[header].named_steps['clf'].fit(X, range(len(sentences)))

#         # Predict summary
#         summary_idx = pipelines[header].named_steps['clf'].predict(X)[0]
#         summary = sentences[summary_idx]
        
#         # Add summary to summaries dict
#         summaries[header] = summary
        
#     print("-------------------------")
#     print(summaries)
#     return JsonResponse({'summaries': summaries})



# # Set up classifiers


# naive_bayes = MultinomialNB()
# neural_network = MLPClassifier(hidden_layer_sizes=(10, 10))
# decision_tree = DecisionTreeClassifier()

# def summarize(request):
#     # Get text from POST request
#     text = request.POST.get('text', '')
#     print("-------------------------")
#     print(text)
#     # Preprocess text
#     sentences = nltk.sent_tokenize(text)
#     print("-------------------------")
#     print(sentences)
#     # Vectorize sentences
#     vectorizer = TfidfVectorizer(stop_words='english')
#     X = vectorizer.fit_transform(sentences)

#     # Feature selection
#     k = 100  # number of features to select
#     selector = SelectKBest(chi2, k=k)
#     X = selector.fit_transform(X, range(len(sentences)))

#     # Get selected mode
#     mode = request.POST.get('mode')
#     print("-------------------------")
#     print(mode)
#     # Set up classifier based on mode and fit with data
#     if mode == 'naive_bayes':
#         classifier = Pipeline([('selector', selector), ('naive_bayes', naive_bayes)])
#         classifier.fit(X, range(len(sentences)))
#     elif mode == 'neural_network':
#         classifier = Pipeline([('selector', selector), ('neural_network', neural_network)])
#         classifier.fit(X, range(len(sentences)))
#     elif mode == 'decision_tree':
#         classifier = Pipeline([('selector', selector), ('decision_tree', decision_tree)])
#         classifier.fit(X, range(len(sentences)))
#     else:
#         # Default to naive bayes if mode is not recognized
#         classifier = Pipeline([('selector', selector), ('naive_bayes', naive_bayes)])
#         classifier.fit(X, range(len(sentences)))

#     # Predict summary using selected classifier
#     summary_idx = classifier.predict(X)[0]
#     summary = sentences[summary_idx]
#     print("-------------------------")
#     print(summary)
#     return JsonResponse({'summary': summary})

# from rouge import Rouge
# from sklearn.model_selection import GridSearchCV, KFold

# naive_bayes = MultinomialNB()
# neural_network = MLPClassifier()
# decision_tree = DecisionTreeClassifier()

# def summarize(request):
#     # Get text from POST request
#     text = request.POST.get('text', '')
#     # Preprocess text
#     sentences = nltk.sent_tokenize(text)
#     # Vectorize sentences
#     vectorizer = TfidfVectorizer(stop_words='english')
#     X = vectorizer.fit_transform(sentences)

#     # Feature selection
#     k = 10  # number of features to select
#     selector = SelectKBest(chi2, k=k)
#     X = selector.fit_transform(X, range(len(sentences)))

#     # Get selected mode
#     mode = request.POST.get('mode')
#     # Set up classifier based on mode
#     if mode == 'naive_bayes':
#         classifier = Pipeline([('selector', selector), ('naive_bayes', naive_bayes)])
#         param_grid = {'naive_bayes__alpha': [0.01, 0.1, 1, 10, 100]}

#     elif mode == 'neural_network':
#         classifier = Pipeline([('selector', selector), ('neural_network', neural_network)])
#         param_grid = {'neural_network__hidden_layer_sizes': [(10,), (20,), (10, 10)]}
#     elif mode == 'decision_tree':
#         classifier = Pipeline([('selector', selector), ('decision_tree', decision_tree)])
#         param_grid = {'decision_tree__max_depth': [None, 10, 20], 'decision_tree__min_samples_leaf': [1, 2, 5]}
#     else:
#         # Default to naive bayes if mode is not recognized
#         classifier = Pipeline([('selector', selector), ('naive_bayes', naive_bayes)])
#         param_grid = {'naive_bayes__alpha': [0.5, 1.0, 1.5]}


#     # Tune hyperparameters using grid search
#     kf = KFold(n_splits=3, shuffle=True, random_state=42) # change the number of splits to 3 or less
#     grid_search = GridSearchCV(classifier, param_grid=param_grid, cv=kf, scoring='accuracy')
#     grid_search.fit(X, range(len(sentences)))
#     best_estimator = grid_search.best_estimator_

#     # Evaluate performance using ROUGE
#     summary_idx = best_estimator.predict(X)[0]
#     summary = sentences[summary_idx]
#     reference_summary = " ".join(sentences)
#     rouge = Rouge()
#     scores = rouge.get_scores(summary, reference_summary)[0]
#     rouge_1_recall = scores['rouge-1']['r']
#     rouge_2_recall = scores['rouge-2']['r']
#     rouge_l_recall = scores['rouge-l']['r']
    
#     # Return summary and ROUGE scores as JSON response
#     return JsonResponse({
#         'summary': summary,
#         'rouge-1-recall': rouge_1_recall,
#         'rouge-2-recall': rouge_2_recall,
#         'rouge-l-recall': rouge_l_recall
#     })

# import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.neural_network import MLPClassifier
# from sklearn.tree import DecisionTreeClassifier

# # Load the dataset files
# train_df = pd.read_csv(r"C:\Users\wuton\OneDrive\Desktop\UNIMAS\FYP\FYP2\Implementation\WebBasedArticleSummarizationWithMachineLearningTechniques\ArticleSummarization\static\dataset\cnn_dailymail\train.csv", nrows=10)
# texts = train_df["article"].tolist()
# labels = train_df["highlights"].tolist()

# # Train a Naive Bayes model on the given texts
# def train_model(texts, labels):
#     vectorizer = CountVectorizer(stop_words="english")
#     X = vectorizer.fit_transform(texts)
#     clf = MultinomialNB()
#     clf.fit(X, labels)
#     return vectorizer, clf

# # Train a Neural Network model on the given texts
# def train_neural_network(texts, labels):
#     vectorizer = CountVectorizer(stop_words="english")
#     X = vectorizer.fit_transform(texts)
#     clf = MLPClassifier(hidden_layer_sizes=(50, 50))
#     clf.fit(X, labels)
#     return vectorizer, clf

# # Train a Decision Tree model on the given texts
# def train_decision_tree(texts, labels):
#     vectorizer = CountVectorizer(stop_words="english")
#     X = vectorizer.fit_transform(texts)
#     clf = DecisionTreeClassifier()
#     clf.fit(X, labels)
#     return vectorizer, clf

# # Classify a new text using the given vectorizer and classifier
# def classify_text(text, vectorizer, clf):
#     X = vectorizer.transform([text])
#     y = clf.predict(X)
#     return y[0]

# def summarize(request):
#     text = request.POST.get('text', '')

#     # Train and classify using Naive Bayes
#     vectorizer_nb, clf_nb = train_model(texts, labels)
#     classification_nb = classify_text(text, vectorizer_nb, clf_nb)

#     # Train and classify using Neural Network
#     vectorizer_nn, clf_nn = train_neural_network(texts, labels)
#     classification_nn = classify_text(text, vectorizer_nn, clf_nn)

#     # Train and classify using Decision Tree
#     vectorizer_dt, clf_dt = train_decision_tree(texts, labels)
#     classification_dt = classify_text(text, vectorizer_dt, clf_dt)

#     return JsonResponse({'summary': classification_nb, 'neural_network': classification_nn, 'decision_tree': classification_dt})
