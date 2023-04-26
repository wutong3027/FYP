import openai
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Set up OpenAI API key
openai.api_key = "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"

# Train a Naive Bayes model on the given texts
def train_model(texts, labels):
    vectorizer = CountVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)
    clf = MultinomialNB()
    clf.fit(X, labels)
    return vectorizer, clf

# Classify a new text using the given vectorizer and classifier
def classify_text(text, vectorizer, clf):
    X = vectorizer.transform([text])
    y = clf.predict(X)
    return y[0]

# Generate a summary of the input text using the OpenAI GPT-3 model
def generate_summary(text):
    prompt = (f"Please summarize the following text in a few sentences:\n\n{re.sub('[^0-9a-zA-Z\n\.\'\",!?]+', ' ', text)}")
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=0.5,
        max_tokens=60,
        n=1,
        stop=None,
        timeout=10,
    )
    summary = response.choices[0].text.strip()
    return summary

# Example usage
texts = [
    "I love this sandwich.",
    "This is an amazing place!",
    "I feel very good about these beers.",
    "This is my best work.",
    "What an awesome view",
    "I do not like this restaurant",
    "I am tired of this stuff.",
    "I can't deal with this",
    "He is my sworn enemy!",
    "My boss is horrible.",
]
labels = ["positive", "positive", "positive", "positive", "positive", "negative", "negative", "negative", "negative", "negative"]

vectorizer, clf = train_model(texts, labels)

input_text = "I had a terrible experience at the restaurant. The food was bad and the service was even worse."
classification = classify_text(input_text, vectorizer, clf)

summary = generate_summary(input_text)

print("Input text:", input_text)
print("Classification:", classification)
print("Summary:", summary)
