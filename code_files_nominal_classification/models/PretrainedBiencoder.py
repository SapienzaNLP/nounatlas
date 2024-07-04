from sentence_transformers import SentenceTransformer, util

class PretrainedBiencoder:
    def __init__(self, model_name = "sentence-transformers/all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, sentences):
        embeddings = self.model.encode(sentences, convert_to_tensor=True)
        return embeddings

    def compute_similarity(self, sentence1, sentence2):
        embedding1 = self.encode([sentence1])[0]
        embedding2 = self.encode([sentence2])[0]
        similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
        return similarity
    
    @staticmethod
    def compute_embedding_similarity(embedding1, embedding2):
        similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
        return similarity