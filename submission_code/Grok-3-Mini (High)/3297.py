class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        N = len(word)
        for t in range(1, N // k + 1):
            L = N - k * t
            if word[:L] == word[N - L:N]:
                return t