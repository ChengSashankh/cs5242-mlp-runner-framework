# Read and prepare the data: From Tran's code
import zipfile

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from analytics.analytics import Analytics
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

    def read(self, path_to_zip):
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

        print(df['genre'].value_counts())
        return self._prepare_dataset(df)

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

        # Transform genre into one hot encoding
        # Y = torch.zeros((df.shape[0], len(FILMS_GENRE)))
        # targets_encoded = df['genre'].apply(lambda g: TARGET_LKP[g])
        # for i, tgt in enumerate(targets_encoded):
        #     Y[i, tgt] = 1

        Y = torch.tensor(np.array(df['genre'].apply(lambda g: TARGET_LKP[g])))

        self.logger.log(f"Shape of X: {X.shape}")
        self.logger.log(f"Shape of Y: {Y.shape}")

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

        Analytics.show_train_val_test_stats(y_train, y_val, y_test)

        self.logger.log(f"Training set: {X_train.shape}, validation set: {X_val.shape}, test set: {X_test.shape}")

        return X_train, X_test, y_train, y_test, X_val, y_val
