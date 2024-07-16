from enum import Enum, auto
from pathlib import Path
from pandas import DataFrame, concat, read_csv
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


def vectorize_txt(tweet_data: DataFrame) -> DataFrame:
    """
    Vectorizes the text column in the dataframe using TF-IDF and adds the resulting tokens as new columns.
    """
    vectorizer = TfidfVectorizer()
    X_txt: spmatrix = vectorizer.fit_transform(tweet_data["cleaned_txt"])
    tfidf_df = DataFrame(X_txt.toarray(), columns=vectorizer.get_feature_names_out())
    tweet_data = concat(
        [tweet_data.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1
    )
    tweet_data.drop(columns=["cleaned_txt","tokenized_text"], inplace=True)
    print(tweet_data.columns)
    return tweet_data


class Model(str, Enum):
    """Different used models"""
    LOGISTIC_REGRESSION = auto()
    DECISION_TREE = auto()
    RANDOM_FOREST = auto()
    SUPPORT_VECTOR_MACHINE = auto()
    XGBOOST = auto()

MODELS = {
    Model.LOGISTIC_REGRESSION: LogisticRegression(),
    Model.DECISION_TREE: DecisionTreeClassifier(),
    Model.RANDOM_FOREST: RandomForestClassifier(n_estimators=100),
    Model.SUPPORT_VECTOR_MACHINE: SVC(),
    Model.XGBOOST: XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
}

def compare_models_on_df(tweet_data: DataFrame) -> DataFrame:
    """Selects all the models in MODELS and compares their results on tweet_data."""
    VALUES = tweet_data.copy()
    VALUES.drop(columns=["target"], inplace= True)
    PREDICT = tweet_data["target"]
    PREDICT.columns = ["target","target_null"]
    PREDICT.drop(columns=["target_null"], inplace= True)

    X_train, X_test, y_train, y_test = train_test_split(
        VALUES, PREDICT, test_size=0.3, random_state=42
    )

    results = {}
    for model_name, model_instance in MODELS.items():
        try:
            model_instance.fit(X_train, y_train)
            y_pred = model_instance.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[model_name.name] = accuracy
        except Exception as e:
            print(f"Failed to use {model_name} due to {e}")
            continue
    results_df = DataFrame(results.items(), columns=["Model", "Accuracy"])
    return results_df

def compare_models_on_csv(tweet_path: Path | str) -> None:
    tweet_data = read_csv(tweet_path)
    vectorized_data = vectorize_txt(tweet_data)
    compare = compare_models_on_df(vectorized_data)
    print(compare)
