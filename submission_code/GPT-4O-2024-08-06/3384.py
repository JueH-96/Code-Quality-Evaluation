class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        num_blocks = n // k
        operations = 0
        
        # For each position in the block of length k
        for i in range(k):
            # Count frequency of each character at this position across all blocks
            freq = {}
            for j in range(num_blocks):
                char = word[j * k + i]
                if char in freq:
                    freq[char] += 1
                else:
                    freq[char] = 1
            
            # Find the maximum frequency of any character at this position
            max_freq = max(freq.values())
            
            # The number of operations needed for this position is the number of blocks
            # minus the maximum frequency of any character at this position
            operations += (num_blocks - max_freq)
        
        return operations