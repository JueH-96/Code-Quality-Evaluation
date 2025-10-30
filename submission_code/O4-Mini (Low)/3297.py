class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        # We try t from 1 up to the smallest t0 where t0*k >= n.
        # For t where t*k < n, the suffix of word[t*k:] must match the prefix of length n - t*k.
        # If we find such t, that's the minimum. Otherwise, once t*k >= n,
        # no original chars remain and we can rebuild the entire word by our choices,
        # so we return the first such t, namely ceil(n/k).
        import math
        t0 = math.ceil(n / k)
        for t in range(1, t0):
            rm = t * k
            # rm < n here
            if word[rm:] == word[:n - rm]:
                return t
        return t0