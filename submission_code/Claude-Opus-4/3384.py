class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        from collections import Counter
        
        n = len(word)
        blocks = []
        
        # Extract all blocks of length k at positions divisible by k
        for i in range(0, n, k):
            blocks.append(word[i:i+k])
        
        # Count frequency of each block
        block_count = Counter(blocks)
        
        # Find the most frequent block
        max_frequency = max(block_count.values())
        
        # Minimum operations = total blocks - most frequent block count
        return len(blocks) - max_frequency