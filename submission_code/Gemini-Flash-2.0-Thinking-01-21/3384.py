class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        num_blocks = n // k
        block_counts = {}
        for i in range(0, n, k):
            block = word[i:i+k]
            block_counts[block] = block_counts.get(block, 0) + 1
        max_frequency = 0
        for block in block_counts:
            if block_counts[block] > max_frequency:
                max_frequency = block_counts[block]
        return num_blocks - max_frequency