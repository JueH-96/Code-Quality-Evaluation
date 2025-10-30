class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        m = n // k  # Number of blocks
        block_freq = {}
        for i in range(m):
            block_start = i * k
            block = word[block_start:block_start + k]
            if block in block_freq:
                block_freq[block] += 1
            else:
                block_freq[block] = 1
        max_freq = max(block_freq.values())
        return m - max_freq