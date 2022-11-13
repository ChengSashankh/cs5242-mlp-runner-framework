import os

import torch
import gensim.downloader as api

from base.constants import OUTPUT_DIR, LOGFILE
from log.logger import Logger


class EmbeddingLoader:
    def __init__(self, embedding_name: str, logger):
        self.logger = logger
        self.embeddings = api.load(embedding_name)

    def embed(self, sentence: []):
        valid_words = [word for word in sentence if word in self.embeddings.key_to_index]
        embedded = torch.tensor([self.embeddings[word] for word in valid_words])
        hit_rate = len(valid_words)/len(sentence)
        self.logger.log (f"Dimension of embedded array: {embedded.shape}, hit rate: {hit_rate}")
        return embedded, hit_rate

    def get_mean(self, sentence: []):
        valid_words = [word for word in sentence if word in self.embeddings.key_to_index]
        return self.embeddings.get_mean_vector(valid_words)


if __name__ == "__main__":
    loader = EmbeddingLoader("glove-twitter-25")
    embedded, hit_rate = loader.embed("hello Sashankh".split(" "))
    print (loader.get_mean("hello hello".split(" ")))
    print (embedded)
    print (loader.embed(["hello"])[0])
    print (hit_rate)










