import faiss


class ExampleIndex():
    def __init__(self, vectors, labels):
        self.dimension = vectors.shape[1]
        self.vectors = vectors.astype('float32')
        self.labels = labels    
   
    def build(self):
        self.index = faiss.IndexFlatL2(self.dimension,)
        self.index.add(self.vectors)
        
    def query(self, vectors, k=10):
        distances, indices = self.index.search(vectors, k) 
        return [self.labels[i] for i in indices[0]], distances[0]