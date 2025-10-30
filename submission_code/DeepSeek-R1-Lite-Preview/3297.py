class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        for t in range(1, n + 1):
            m = t * k % n
            if word[m:] + word[:m] == word:
                return t
        return n  # Fallback, though t <= n should cover all cases