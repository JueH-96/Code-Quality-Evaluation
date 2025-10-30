class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        num_blocks = n // k
        from collections import defaultdict
        block_count = defaultdict(int)
        
        for i in range(num_blocks):
            block = word[i * k : (i + 1) * k]
            block_tuple = tuple(block)
            block_count[block_tuple] += 1
        
        max_freq = max(block_count.values(), default=0)
        return num_blocks - max_freq