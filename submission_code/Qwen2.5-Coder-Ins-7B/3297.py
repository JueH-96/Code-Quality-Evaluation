class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        for t in range(1, n // k + 1):
            if word.startswith(word[t * k:]):
                return t
        return n // k + 1