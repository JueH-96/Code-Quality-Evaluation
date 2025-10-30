class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        for i in range(1, n + 1):  # Consider a max of n operations
            offset = (i * k) % n
            if word[offset:] == word[0:n - offset]:
                return i
        return n  # Fallback, should not be reached for valid inputs