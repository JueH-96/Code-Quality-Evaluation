class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        m = n // k
        # Count how many times each length-k block appears
        count = {}
        for i in range(0, n, k):
            block = word[i:i+k]
            count[block] = count.get(block, 0) + 1
        # We pick the most frequent block as our target pattern
        max_freq = max(count.values())
        # We must change every other block to that pattern
        return m - max_freq