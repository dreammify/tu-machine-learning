import pandas
from sklearn.feature_extraction.text import TfidfVectorizer


def generate_idf_vectors(dataframe: pandas.DataFrame, mask, column: str):
    train_frame = dataframe[mask]
    test_frame = dataframe[~mask]

    tokenizer = TfidfVectorizer(strip_accents='unicode', min_df=10)
    tokenizer.fit(train_frame[column])

    train_tokens = tokenizer.transform(train_frame[column])
    test_tokens = tokenizer.transform(test_frame[column])
    return train_tokens, test_tokens