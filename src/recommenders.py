import numpy as np
import scipy.sparse as sp


def _topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Return indices of top-k scores in descending order using argpartition."""
    k = min(k, len(scores))
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.argsort(-scores[part])]


class PopularityRecommender:
    """Ranks candidates by global training frequency — context-unaware baseline."""

    def __init__(self, item_counts: np.ndarray):
        self._scores = item_counts.astype(np.float64)

    def recommend(self, observed: set[int], candidates: set[int], k: int) -> list[int]:
        cand_arr = np.fromiter(candidates, dtype=np.intp, count=len(candidates))
        cand_scores = self._scores[cand_arr]
        return cand_arr[_topk(cand_scores, k)].tolist()


class LocalAggRecommender:
    """Averages direct confidence weights from observed items to each candidate."""

    def __init__(self, W: sp.csr_matrix):
        self._W = W

    def recommend(self, observed: set[int], candidates: set[int], k: int) -> list[int]:
        obs_list = list(observed)
        scores = np.array(self._W[obs_list, :].sum(axis=0)).flatten() / len(obs_list)
        cand_arr = np.fromiter(candidates, dtype=np.intp, count=len(candidates))
        cand_scores = scores[cand_arr]
        return cand_arr[_topk(cand_scores, k)].tolist()


class PPRRecommender:
    """
    Personalized PageRank over the confidence graph.

    Observed products seed the initial relevance; scores propagate along
    directed edges via power iteration. Captures indirect associations.
    W_norm must be row-stochastic (precomputed outside this class).
    """

    def __init__(
        self,
        W_norm: sp.csr_matrix,
        alpha: float = 0.85,
        max_iter: int = 50,
        tol: float = 1e-6,
    ):
        self._W_norm = W_norm.T.tocsr()  # pre-transpose once
        self._alpha = alpha
        self._max_iter = max_iter
        self._tol = tol

    def recommend(self, observed: set[int], candidates: set[int], k: int) -> list[int]:
        n = self._W_norm.shape[0]
        obs_list = list(observed)

        r0 = np.zeros(n)
        r0[obs_list] = 1.0 / len(obs_list)
        r = r0.copy()

        for _ in range(self._max_iter):
            r_new = self._alpha * self._W_norm.dot(r) + (1 - self._alpha) * r0
            if np.abs(r_new - r).sum() < self._tol:
                r = r_new
                break
            r = r_new

        r[obs_list] = 0.0
        cand_arr = np.fromiter(candidates, dtype=np.intp, count=len(candidates))
        return cand_arr[_topk(r[cand_arr], k)].tolist()

    def recommend_batch(
        self,
        observed_list: list[set[int]],
        candidates_arr: np.ndarray,
        k: int,
    ) -> list[list[int]]:
        """
        Batch PPR for B observed sets in one pass.

        Stacks seed vectors as columns and runs a single sparse matmul per
        iteration instead of B separate matvecs. Observed items are zeroed to
        -inf before top-k so they never appear in recommendations.
        """
        n = self._W_norm.shape[0]
        B = len(observed_list)

        R0 = np.zeros((n, B))
        for j, obs in enumerate(observed_list):
            obs_arr = list(obs)
            R0[obs_arr, j] = 1.0 / len(obs_arr)

        R = R0.copy()
        for _ in range(self._max_iter):
            R_new = self._alpha * self._W_norm.dot(R) + (1 - self._alpha) * R0
            if np.abs(R_new - R).sum(axis=0).max() < self._tol:
                R = R_new
                break
            R = R_new

        results = []
        for j, obs in enumerate(observed_list):
            r = R[:, j].copy()
            r[list(obs)] = -np.inf
            results.append(candidates_arr[_topk(r[candidates_arr], k)].tolist())
        return results
