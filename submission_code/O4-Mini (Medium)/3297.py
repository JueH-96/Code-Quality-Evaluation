class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        # We look for the smallest t > 0 such that after t operations
        # the window shifted by t*k recovers the original word.
        # Model: we build an infinite tape Z starting with `word`, then
        # appending our chosen blocks X_0, X_1, ... of length k.
        # After t steps the current window is Z[t*k : t*k + n].
        # We want Z[t*k : t*k + n] == word.  For positions p in [t*k, n-1],
        # Z[p] is fixed = word[p], and must equal word[p - t*k].
        # That yields the check: if t*k < n,       word[t*k:] == word[:n - t*k]
        # otherwise (t*k >= n) we can choose all appended to match, so it's
        # automatically possible.
        #
        # We just try t = 1,2,... until either t*k >= n or that suffix-prefix
        # match holds.
        
        # maximum t we need to test before t*k >= n
        max_t = (n + k - 1) // k
        for t in range(1, max_t + 1):
            if t * k >= n:
                return t
            # check if suffix of length n - t*k equals prefix of same length
            if word[t*k:] == word[:n - t*k]:
                return t
        # in principle we always return inside the loop
        return max_t