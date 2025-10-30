class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        num_blocks = n // k
        blocks = []
        for i in range(num_blocks):
            blocks.append(word[i * k : (i + 1) * k])
        block_counts = {}
        for block in blocks:
            block_counts[block] = block_counts.get(block, 0) + 1
        max_frequency = 0
        for block in block_counts:
            max_frequency = max(max_frequency, block_counts[block])
        return num_blocks - max_frequency