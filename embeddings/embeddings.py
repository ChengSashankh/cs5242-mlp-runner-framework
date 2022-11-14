import numpy as np
import torch
import gensim.downloader as api
from sklearn.feature_extraction.text import TfidfVectorizer

from log.logger import Logger


class EmbeddingLoader:
    def __init__(self, embedding_name: str, logger):
        self.logger = logger
        self.embeddings = api.load(embedding_name)

    def embed(self, sentence: []):
        valid_words = [word for word in sentence if word in self.embeddings.key_to_index]
        # embedded = [self.embeddings[word] for word in valid_words]
        embedded = []
        for word in sentence:
            if word in self.embeddings.key_to_index:
                embedded.append(self.embeddings[word])
            else:
                embedded.append(np.zeros_like(self.embeddings['hello']))

        embedded = np.array(embedded)
        hit_rate = len(valid_words)/len(sentence)
        self.logger.log (f"Dimension of embedded array: {embedded.shape}, hit rate: {hit_rate}")
        return embedded, hit_rate

    def get_mean(self, sentence: []):
        valid_words = [word for word in sentence if word in self.embeddings.key_to_index]
        hit_rate = len(valid_words)/len(sentence)
        self.logger.log(f"Hit rate: {hit_rate}")
        return self.embeddings.get_mean_vector(valid_words)

    def get_tf_idf_weighted_mean(self, all_plots: [], tf_idf_vecs):
        vec_rep = []
        total_hit_rate = 0
        for i, plot in enumerate(all_plots):
            scores = tf_idf_vecs[i, :len(plot)]
            vectors, hit_rate = self.embed(plot)
            total_hit_rate += hit_rate
            self.logger.log(f"Hit rate: {hit_rate}")
            self.logger.log(f"Vectors shape: {vectors.shape}")

            n_vecs = vectors.shape[0]
            assert len(scores) == n_vecs
            weighted_mean = torch.zeros(vectors[0].shape)
            for j in range(n_vecs):
                weighted_mean = weighted_mean + (scores[j] * vectors[j])

            vec_rep.append(np.array(weighted_mean))

        self.logger.log(f"Overall hit rate: {total_hit_rate/len(all_plots)}")
        return torch.tensor(np.array(vec_rep))


if __name__ == "__main__":
    loader = EmbeddingLoader("glove-twitter-25", Logger(".", "test", "test"))
    _embedded, _hit_rate = loader.embed("hello Sashankh".split(" "))
    print (loader.get_mean("hello hello".split(" ")))
    print (_embedded)
    print (loader.embed(["hello"])[0])
    print (_hit_rate)










