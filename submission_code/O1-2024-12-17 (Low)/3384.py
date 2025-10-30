class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        # n is the length of the string
        n = len(word)
        
        # Number of blocks of length k
        num_blocks = n // k
        
        # Count the occurrence of each block of length k
        from collections import Counter
        block_counter = Counter(word[i:i+k] for i in range(0, n, k))
        
        # The best block to unify everything to is the one with the highest frequency
        max_freq = max(block_counter.values())
        
        # Minimum operations needed is the total number of blocks minus the highest frequency
        return num_blocks - max_freq