class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        for t in range(1, n + 1):
            if all(word[(i * k) % n] == word[i] for i in range(n)):
                return t
        return n