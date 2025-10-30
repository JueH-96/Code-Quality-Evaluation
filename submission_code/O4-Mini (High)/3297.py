class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        # the smallest t such that t*k >= n is always feasible
        max_t = (n + k - 1) // k
        for t in range(1, max_t + 1):
            x = n - t * k
            # if x <= 0 then t*k >= n, or if the suffix of length x equals the prefix of length x
            if x <= 0 or word[:x] == word[t * k:]:
                return t