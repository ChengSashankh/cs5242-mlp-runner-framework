import os
import zipfile

import pandas as pd
import torch
from torch.utils.data import Dataset

from base.constants import LOGFILE, OUTPUT_DIR
from log.logger import Logger


class FilmGenres(Dataset):
    def __init__(self, zip_file_path: str):
        """
        train: True if dataset for training, otherwise False is for testing
        zip_file_path is path containing the plot films
        """
        # Reading from zip files, and storing in a dictionary with key is fim genre, and value is list of films with plot
        initial_df = pd.DataFrame()
        self.logger = Logger(os.environ["MLP_OUTPUT_DIR"], LOGFILE, FilmGenres.__name__)

        with zipfile.ZipFile(zip_file_path) as z:
            for name in z.namelist():
                if name.endswith(".csv"):
                    self.logger.log(f'Loading data from {name}...')
                    x = pd.read_csv(z.open(name))
                    self.logger.log(f'Loading completed from {name}...')
                    initial_df = pd.concat([initial_df, x[['genre', 'plot']]], axis=0, ignore_index=True)
            self.logger.log("Dataframe (df) ready to be used!")

        self.df = initial_df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx >= len(self):
            raise IndexError

        sample = self.df.loc[idx, "plot"], self.df.loc[idx, "genre"]

        return sample
