class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        block_count = {}
        
        # Count how many times each k-length block appears
        for i in range(0, n, k):
            block = word[i:i+k]
            block_count[block] = block_count.get(block, 0) + 1
        
        # Find the block that appears the most
        max_freq = max(block_count.values())
        
        # The minimum operations needed is total blocks minus the most frequent block's count
        return (n // k) - max_freq