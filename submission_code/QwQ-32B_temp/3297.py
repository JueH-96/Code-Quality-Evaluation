class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        max_t = (n + k - 1) // k  # Equivalent to ceiling(n/k)
        for t in range(1, max_t + 1):
            pos = t * k
            if word[pos:] == word[:n - pos]:
                return t
        return max_t  # This line is theoretically unreachable but added for safety