class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        # We try times T from 1 up to n.
        # Condition:
        # 1) If T*k >= n, we can already rebuild the entire string by choosing
        #    all appended characters to match the original (since we've removed
        #    all original chars).
        # 2) Otherwise, check if the last (n - T*k) characters of the original
        #    can match its first (n - T*k) characters.  Concretely:
        #       word[T*k:] == word[:n - T*k]
        #    If so, we can choose the appended substring(s) to rebuild the rest.
        for T in range(1, n + 1):
            if T * k >= n:
                return T
            # Check if the "tail" of length (n - T*k) matches the "head" of length (n - T*k).
            if word[T * k:] == word[: n - T * k]:
                return T

        # The loop always returns by T = n, so we never actually get here.
        return n