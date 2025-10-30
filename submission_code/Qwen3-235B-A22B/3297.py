class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        max_t = (n + k - 1) // k
        for t in range(1, max_t + 1):
            m = k * t
            substr = word[m:]
            if word.startswith(substr):
                return t
        return max_t  # This line is theoretically unreachable