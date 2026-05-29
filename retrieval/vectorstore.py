import json
import faiss
import numpy as np

from retrieval.embeddings import embed


class VectorStore:

    def __init__(self):

        self.index = None
        self.patterns = []

        self.load()

    def load(self):

        with open(
            "retrieval/patterns.json",
            "r"
        ) as f:

            self.patterns = json.load(f)

        vectors = np.array(
            [
                embed(
                    p.get(
                        "embed_text",
                        f"{p.get('title','')} {p.get('description','')}"
                    )
                )
                for p in self.patterns
            ]
        ).astype("float32")

        dim = vectors.shape[1]

        self.index = faiss.IndexFlatL2(dim)

        self.index.add(vectors)

    def query_batch(
        self,
        queries,
        top_k=5
    ):

        results = []

        for query in queries:

            vec = np.array(
                [embed(query)]
            ).astype("float32")

            distances, indices = self.index.search(
                vec,
                top_k
            )

            for idx, score in zip(
                indices[0],
                distances[0]
            ):

                item = dict(
                    self.patterns[idx]
                )

                item["pattern_id"] = item["id"]
                item["score"] = float(score)

                results.append(item)

        return results