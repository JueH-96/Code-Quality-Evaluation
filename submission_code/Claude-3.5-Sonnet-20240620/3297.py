class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        for i in range(1, n // k + 1):
            if word[i*k:] == word[:n-i*k]:
                return i
        return (n + k - 1) // k