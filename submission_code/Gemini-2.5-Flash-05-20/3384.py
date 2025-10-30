import collections

class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        
        # Calculate the total number of blocks of length k
        # Since k divides n, n // k gives the exact number of blocks.
        num_blocks = n // k
        
        # Use collections.Counter to efficiently count the occurrences of each unique k-length block.
        block_counts = collections.Counter()
        
        # Iterate through the word to extract each k-length block.
        # The loop steps by k to get the starting index of each block.
        for i in range(0, n, k):
            block = word[i : i + k] # Extract the substring (block) of length k
            block_counts[block] += 1 # Increment its count
            
        # Find the maximum frequency among all blocks.
        # If block_counts is empty (which won't happen based on constraints as n >= 1), max_freq would be 0.
        # However, since n >= 1 and k <= n, num_blocks will be at least 1, so block_counts will have at least one entry.
        max_freq = 0
        if block_counts:
            max_freq = max(block_counts.values())
        
        # The minimum number of operations is the total number of blocks minus the count of the most frequent block.
        # This is because we aim to change all blocks to match the most frequent one.
        # The 'max_freq' blocks already match and require no operation.
        # The remaining (num_blocks - max_freq) blocks each require one operation to be changed.
        return num_blocks - max_freq