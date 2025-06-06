import pyterrier as pt
import numpy as np
import string


class W2VExpander(pt.transformer.Transformer):
    def __init__(
        self, model, method="centroid", n_similar=10, nu=10, interpolate_lambda=0.0
    ):
        super().__init__()
        self.model = model
        self.method = method
        self.n_similar = n_similar
        self.nu = nu
        self.interpolate_lambda = interpolate_lambda

        # Prepare vocabulary keys and normalized vectors for fast lookup
        if hasattr(model, "get_normed_vectors"):
            self.keys = model.index_to_key
            self.vects = model.get_normed_vectors()  # already L2-normalized
        else:
            self.keys = list(model.key_to_index.keys())
            vects = np.vstack([model[w] for w in self.keys])
            norms = np.linalg.norm(vects, axis=1, keepdims=True)
            self.vects = vects / np.where(norms == 0, 1, norms)

        self.key_idx = {w: i for i, w in enumerate(self.keys)}

    def transform(self, topics):
        def _expand(q):
            tokens = [w for w in q.split() if w in self.key_idx]
            if not tokens:
                return q

            # Stack and normalize query token vectors
            Q = np.vstack([self.model[w] for w in tokens])
            norms = np.linalg.norm(Q, axis=1, keepdims=True)
            Q = Q / np.where(norms == 0, 1, norms)

            V = len(self.keys)
            # Compute raw method scores over the vocab
            if self.method == "centroid":
                centroid = Q.mean(axis=0)
                raw = centroid.dot(self.vects.T)
            else:
                sims = Q.dot(self.vects.T)  # (n_tokens, V)
                raw = np.zeros(V, dtype=float)
                for i, sim_row in enumerate(sims):
                    # pick the top-n_similar indices for this token
                    top_idx = np.argpartition(-sim_row, self.n_similar - 1)[
                        : self.n_similar
                    ]
                    # compute a softmax *over only those*
                    exp_s = np.exp(sim_row[top_idx])
                    p = exp_s / exp_s.sum()
                    # accumulate according to the chosen Comb method
                    if self.method == "combSUM":
                        raw[top_idx] += p
                    elif self.method == "combMNZ":
                        raw[top_idx] += p * 1  # we'll multiply by count below
                    elif self.method == "combMAX":
                        raw[top_idx] = np.maximum(raw[top_idx], p)
                if self.method == "combMNZ":
                    # multiply by the number of times each term got a non-zero
                    counts = (raw > 0).astype(int)
                    raw = raw * counts

            # Sum-normalize raw scores to form a distribution
            raw = np.where(raw < 0, 0, raw)  # ensure non-negative
            total = raw.sum() or 1.0
            scores = raw / total

            # Interpolate with MLE of original query tokens (for all methods)
            if 0 < self.interpolate_lambda < 1:
                mle = np.zeros_like(scores)
                L = len(tokens)
                for w in tokens:
                    mle[self.key_idx[w]] = 1.0 / L
                scores = (
                    1 - self.interpolate_lambda
                ) * scores + self.interpolate_lambda * mle

            # Mask out original tokens
            mask = np.ones(V, dtype=bool)
            for w in tokens:
                mask[self.key_idx[w]] = False

            # Select top-nu candidates
            nu = min(self.nu, mask.sum())
            idxs = np.argpartition(-scores[mask], nu - 1)[:nu]
            masked_idxs = np.nonzero(mask)[0]
            chosen = [
                self.keys[i].translate(str.maketrans("", "", string.punctuation))
                for i in masked_idxs[idxs]
            ]

            return q + " " + " ".join(chosen)

        topics = topics.copy()
        topics["query"] = topics["query"].apply(_expand)
        return topics
