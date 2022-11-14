# Read and prepare the data: From Tran's code
import zipfile

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from analytics.analytics import Analytics
from base.TfIdfWeighter import TfIdfWeighter
from base.constants import TARGET_LKP, OUTPUT_DIR, LOGFILE, FILMS_GENRE
from embeddings.embeddings import EmbeddingLoader
from log.logger import Logger
from base.sentence_cleaner import SentCleaner
from utils import get_default_sent_cleaner_conf, get_default_down_sample_conf


##########################################################################
# Dataset Read Logic
##########################################################################
class DatasetReader:
    def __init__(self, embedding_type, logger, sent_cleaner_conf=None, down_sample_conf=None):
        if sent_cleaner_conf is None:
            sent_cleaner_conf = get_default_sent_cleaner_conf()
        if down_sample_conf is None:
            down_sample_conf = get_default_down_sample_conf()

        # Member assignments
        self.logger = logger
        self.embedding_loader = EmbeddingLoader(embedding_type, logger)
        self.sent_cleaner_conf = sent_cleaner_conf
        self.down_sample_conf = down_sample_conf

    def read(self, path_to_zip, simple):
        # Reading from zip files, and storing in a dictionary with key is fim genre,
        # and value is list of films with plot
        initial_df = pd.DataFrame()

        with zipfile.ZipFile(path_to_zip) as z:
            for name in z.namelist():
                if name.endswith(".csv"):
                    self.logger.log(f'Loading data from {name}...')
                    x = pd.read_csv(z.open(name))
                    self.logger.log(f'Loading completed from {name}...')
                    initial_df = pd.concat([initial_df, x[['genre', 'plot']]], axis=0, ignore_index=True)
            self.logger.log("Dataframe (df) ready to be used!")

        df = self._down_sample(initial_df)
        df = df.dropna()

        # Shuffle the df
        df = df.sample(frac=1)

        self.logger.log(df['genre'].value_counts())
        if simple:
            self.logger.log("Preparing vectors using simple average")
            return self._prepare_dataset(df)
        else:
            self.logger.log("Preparing vectors using weighted TF-IDF average")
            return self._prepare_dataset_tfidf(df)

    def _down_sample(self, df):
        # Down sample drama
        for genre in FILMS_GENRE:
            sample_percent = self.down_sample_conf.get(genre, 0.0)
            df = df.drop(df.query(f"genre == '{genre}'").sample(frac=sample_percent).index)

        self.logger.log(df['genre'].value_counts())
        return df

    def _prepare_dataset(self, df):
        # Clean the plots
        cleaned_plots = df['plot'].apply(lambda x: SentCleaner(x, self.sent_cleaner_conf).clean_sent())
        # Create the mean embedding vector
        embed_vectors = torch.tensor(np.array([self.embedding_loader.get_mean(plot) for plot in list(cleaned_plots)]))
        X = embed_vectors
        self.logger.log (f"Mean vector result shape (embed_vectors) for {len(cleaned_plots)} plots: {embed_vectors.shape}")

        Y = torch.tensor(np.array(df['genre'].apply(lambda g: TARGET_LKP[g])))

        self.logger.log(f"Shape of X: {X.shape}")
        self.logger.log(f"Shape of Y: {Y.shape}")

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

        Analytics.show_train_val_test_stats(y_train, y_val, y_test, self.logger)

        return X_train, X_test, y_train, y_test, X_val, y_val

    # TODO: separate tfidf for test set
    def _prepare_dataset_tfidf(self, df):
        # Clean the plots
        cleaned_plots = df['plot'].apply(lambda x: SentCleaner(x, self.sent_cleaner_conf).clean_sent())

        # Calculate TFIDF
        tf_idf_vecs = TfIdfWeighter(cleaned_plots).get_tf_idf()
        embed_vectors = self.embedding_loader.get_tf_idf_weighted_mean(cleaned_plots, tf_idf_vecs)

        X = embed_vectors
        self.logger.log (f"Mean vector result shape (embed_vectors) for {len(cleaned_plots)} plots: {embed_vectors.shape}")
        Y = torch.tensor(np.array(df['genre'].apply(lambda g: TARGET_LKP[g])))

        self.logger.log(f"Shape of X: {X.shape}")
        self.logger.log(f"Shape of Y: {Y.shape}")

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

        Analytics.show_train_val_test_stats(y_train, y_val, y_test, self.logger)

        return X_train, X_test, y_train, y_test, X_val, y_val
