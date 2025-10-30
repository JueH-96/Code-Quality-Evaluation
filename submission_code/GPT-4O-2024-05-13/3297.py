class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        for t in range(1, n + 1):
            if word == word[k*t % n:] + word[:k*t % n]:
                return t
        return n