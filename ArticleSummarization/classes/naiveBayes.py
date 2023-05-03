import re
import heapq
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from ArticleSummarization.classes.machineLearning import MachineLearning

nltk.download('punkt')
nltk.download('stopwords')


class NaiveBayes(MachineLearning):
    def __init__(self):
        super().__init__()
        self.model = MultinomialNB()
        self.tokenizer = CountVectorizer(stop_words='english')

    def NB_generate_summary(self, text):
        num_sentences = 5
        if not isinstance(text, str):
            raise TypeError('Input must be a string')
        if not isinstance(num_sentences, int) or num_sentences < 1:
            raise ValueError('Number of sentences must be a positive integer')
        
        # tokenizing the text by sentence
        sentences = sent_tokenize(text)
        
        processed_sentences = []
        for sentence in sentences:
            words = re.sub('[^a-zA-Z]', ' ', sentence)
            words = words.lower()
            words = words.split()
            ps = PorterStemmer()
            words = [ps.stem(word) for word in words if not word in set(stopwords.words('english'))]
            words = ' '.join(words)
            processed_sentences.append(words)

        # feature extraction
        cv = self.tokenizer
        sentence_features = cv.fit_transform(processed_sentences)
        
        # creating labels for training data
        summary_sentences_idx = heapq.nlargest(num_sentences, range(len(sentences)), key=lambda i: sentence_features[i].sum())
        labels = ['summary' if i in summary_sentences_idx else 'non-summary' for i in range(len(sentences))]
        
        # training the classifier
        clf = self.model
        clf.fit(sentence_features, labels)
        
        # selecting the top 'num_sentences' sentences based on the classifier scores
        sentence_scores = clf.predict_proba(sentence_features)[:, 0]
        summary_sentences_idx = heapq.nlargest(num_sentences, range(len(sentences)), key=lambda i: sentence_scores[i])
        summary_sentences = [sentences[idx] for idx in sorted(summary_sentences_idx)]
        
        # combining the selected sentences to generate the summary
        summary = ' '.join(summary_sentences)
        return summary
