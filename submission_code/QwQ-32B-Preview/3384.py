from typing import Counter

class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        m = n // k  # Number of blocks
        
        # Split the word into blocks of size k
        blocks = [word[i:i+k] for i in range(0, n, k)]
        
        # Count the frequency of each block
        freq = Counter(blocks)
        
        # Find the block with the highest frequency
        max_freq = max(freq.values())
        
        # Calculate the minimum operations needed
        min_operations = m - max_freq
        
        return min_operations