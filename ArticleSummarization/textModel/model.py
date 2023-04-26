import csv
import sys

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Read in the data
csv.field_size_limit(2**30)

def main():
    with open(r"C:\Users\wuton\OneDrive\Desktop\UNIMAS\FYP\FYP2\Implementation\WebBasedArticleSummarizationWithMachineLearningTechniques\ArticleSummarization\static\dataset\archive\scisumm.csv", encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # skip header row
        texts = []
        summaries = []
        for row in reader:
            text, summary = row[0], row[1]
            texts.append(text)
            summaries.append(summary)
            if text is None or summary is None:
                print("None data found!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    print("Number of texts:", len(texts))
    print("Number of summaries:", len(summaries))
    # Split data into train and test sets
    X_train_test, X_val, y_train_test, y_val = train_test_split(texts, summaries, test_size=0.2, random_state=42)

    # Split train set into train and validation sets
    X_train, X_test, y_train, y_test = train_test_split(X_train_test, y_train_test, test_size=0.25, random_state=42)
    print(f"Train set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")

    # Define the pipeline and hyperparameters to tune
    pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())
    parameters = {
        'tfidfvectorizer__ngram_range': [(1, 1), (1, 2), (2, 2)],
        'tfidfvectorizer__max_df': [0.5, 0.75, 1.0],
        'tfidfvectorizer__min_df': [1, 2, 3],
        'multinomialnb__alpha': [0.1, 1.0, 10.0]
    }
    # Use stratified k-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # Perform grid search to find the best hyperparameters
    clf = GridSearchCV(pipeline, parameters, cv=kf)
    clf.fit(X_train, y_train)

    print(f"Best parameters: {clf.best_params_}")
    print(f"Best cross-validation score: {clf.best_score_:.4f}")

    # Test the final model on the test data
    y_pred = clf.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
