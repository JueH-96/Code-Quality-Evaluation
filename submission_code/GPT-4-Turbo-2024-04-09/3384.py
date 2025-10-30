class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        # Number of blocks of length k
        num_blocks = n // k
        
        # To store the frequency of each k-length segment at each position mod k
        from collections import defaultdict
        freq = [defaultdict(int) for _ in range(k)]
        
        # Fill frequency tables
        for i in range(num_blocks):
            for j in range(k):
                segment = word[i*k + j]
                freq[j][segment] += 1
        
        # Calculate the minimum number of changes needed
        min_operations = 0
        for j in range(k):
            # Find the most frequent character at position j in each block
            max_freq = max(freq[j].values())
            # We need to change all other characters to this one
            min_operations += (num_blocks - max_freq)
        
        return min_operations