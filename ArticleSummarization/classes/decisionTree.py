import re
import nltk
from sklearn.tree import DecisionTreeRegressor
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from ArticleSummarization.classes.machineLearning import MachineLearning

class DecisionTree(MachineLearning):

    def __init__(self):
        super().__init__()
        self.model = DecisionTreeRegressor()
        self.tokenizer = CountVectorizer()
    
    def DT_generate_summary(self, text):
        sentences = nltk.sent_tokenize(text)

        num_sentences = (len(sentences) // 2) 
        if num_sentences <= 2:
            num_sentences = 3

        # Preprocess the text
        processed_sentences = []
        for sentence in sentences:
            words = re.sub('[^a-zA-Z]', ' ', sentence)
            words = words.lower()
            words = words.split()
            ps = PorterStemmer()
            words = [ps.stem(word) for word in words if not word in set(stopwords.words('english'))]
            words = ' '.join(words)
            processed_sentences.append(words)

        # Create BoW feature vectors
        vectorizer = self.tokenizer
        X = vectorizer.fit_transform(processed_sentences)

        # Train a decision tree to predict sentence importance
        y = [len(sentence) for sentence in sentences]  # Use sentence length as target variable
        dt = self.model
        dt.fit(X, y)

        # Use the decision tree to predict sentence importance scores
        scores = dt.predict(X)

        # Select the top-scoring sentences as the summary
        summary_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:num_sentences]
        summary_sentences = [sentences[i] for i in summary_indices]
        summary = " ".join(summary_sentences)

        return summary