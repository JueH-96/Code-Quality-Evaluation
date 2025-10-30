class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        m = n // k  # number of blocks
        freq = {}
        # Count frequency of each block of length k
        for i in range(0, n, k):
            block = word[i:i+k]
            freq[block] = freq.get(block, 0) + 1
        # The best we can do is pick the most common block as the period string s
        max_count = max(freq.values())
        # We need to overwrite all other blocks
        return m - max_count