class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        if k == n:
            return 1

        # We need to find the smallest time t such that word is reverted to its initial state
        # This happens when the word shifts by a multiple of the word length
        # So we need to find the smallest t such that t * k % n == 0

        for t in range(1, n + 1):
            if (t * k) % n == 0:
                return t

        return -1  # This line is theoretically unreachable due to the problem constraints