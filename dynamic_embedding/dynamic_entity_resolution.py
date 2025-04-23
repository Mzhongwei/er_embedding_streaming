import faiss
from gensim.similarities.annoy import AnnoyIndexer
import numpy as np



class FaissIndex:
    def __init__(self, model):
        self.filtered_word_to_idx = {}  # Record the mapping of words beginning with “idx__” to the FAISS index.
        self.idx_to_filtered_word = {}  # Record the mapping of FAISS index to word

        dimension = model.vector_size
        self.index = faiss.IndexFlatIP(dimension)
        # self.index = faiss.IndexFlatL2(dimension)

        word_vectors = []
        idx = 0  # FAISS internal index
        for word in model.wv.index_to_key:
            if word.startswith("idx__"):
                print(f"id in model: {word}, id in index: {idx}")
                vector = model.wv[word]
                word_vectors.append(vector)
                self.filtered_word_to_idx[word] = idx
                self.idx_to_filtered_word[idx] = word
                idx += 1
        
        if word_vectors:
            mat = np.array(word_vectors, dtype=np.float32)
            faiss.normalize_L2(mat)
            self.index.add(mat)  # add to FAISS
        

    def get_similar_words(self, query_vector_model, query_word, top_k=5):
        """Query FAISS for similar words starting with 'idx__' and return similarity score"""
        if query_word not in self.filtered_word_to_idx:
            return None

        query_vector = np.array(query_vector_model, dtype=np.float32)
        faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(query_vector, int(top_k)*10)

        similar_words = []
        # FAISS returns the L2 distance, which needs to be converted to similarity
        # cosine sim = 1 / (1 + L2_distance)
        for dist, idx in zip(distances[0], indices[0]) :
            print(f"search id: {idx}")
            similar_word  = self.idx_to_filtered_word[idx]
            # print(similar_word)
            if similar_word != query_word and idx in self.idx_to_filtered_word and float(dist) > 0.5:
            # if similar_word != query_word and idx in self.idx_to_filtered_word:
                # similar_words = [(similar_worcd, 1 / (1 + dist))]  # convert to similarity
                similar_words.append((similar_word,  float(dist)))

        return similar_words


    def update_index(self, model):
        """update FAISS index
        Keep the index of previously trained nodes and add new indexes only for new nodes
        """
        new_words = [w for w in model.wv.index_to_key if w not in self.filtered_word_to_idx and w.startswith("idx__")]
        if not new_words:
            return

        new_vectors = np.array([model.wv[w] for w in new_words], dtype=np.float32)
        faiss.normalize_L2(new_vectors)

        self.index.add(new_vectors)

        # update map
        current_size = len(self.filtered_word_to_idx)
        for i, word in enumerate(new_words):
            self.filtered_word_to_idx[word] = current_size + i
            self.idx_to_filtered_word[current_size + i] = word
    
    def rebuild_index(self, model, id_num):
        '''rebuild the whole index
        update index of previously trained nodes and newly trained nodes
        '''
        self.index.reset()  # clean FAISS index
        self.filtered_word_to_idx.clear()
        self.idx_to_filtered_word.clear()

        w_list=[]
        for i in range(id_num + 1):
            target = f"idx__{i}"
            if target in model.wv.key_to_index:
                w_list.append(target)
            else: 
                raise ValueError(f"Can't find word {target} in embedding model")
        
        v_list = []
        idx = 0  # FAISS internal index
        for word in w_list:
            if word.startswith("idx__"):
                vector = model.wv[word]
                v_list.append(vector)
                self.filtered_word_to_idx[word] = idx
                self.idx_to_filtered_word[idx] = word
                idx += 1

        if v_list:
            mat = np.array(v_list, dtype=np.float32)
            faiss.normalize_L2(mat)
            self.index.add(mat)  # add to FAISS


def dynentity_resolution(model, target, n):
    filtered_keys = [word for word in model.wv.index_to_key if word.startswith("idx__")] # only search words beginning with "idx__"
    sims = []
    sims = [(word, score) for word, score in model.wv.most_similar(target, topn=n*10) if word in filtered_keys and score >0.5][:n]
    # sims = [(word, score) for word, score in model.wv.most_similar(target, topn=n*10) if word in filtered_keys][:n]
    # sims = model.wv.most_similar(target, topn=10, restrict_vocab=len(filtered_keys))  # get other similar words
    return sims   