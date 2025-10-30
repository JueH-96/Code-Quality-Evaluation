import collections # It's standard practice to place imports at the top of the file.

class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        """
        Calculates the minimum number of operations to make the given word k-periodic.

        An operation consists of picking two indices i and j, both divisible by k, 
        and replacing the substring word[i:i+k] with word[j:j+k].
        A word is k-periodic if it's formed by concatenating a string s of length k multiple times.

        Args:
          word: The input string.
          k: The length of the periodic segment. k must divide len(word).

        Returns:
          The minimum number of operations required.
        """
        
        n = len(word)
        
        # Calculate the total number of k-length blocks in the word.
        # Since the problem guarantees that k divides n, n // k gives the exact count.
        m = n // k
        
        # Use collections.Counter to efficiently count the frequency of each unique block.
        # A block is a substring of length k starting at an index divisible by k.
        block_counts = collections.Counter()
        
        # Iterate through the word with a step of k to identify the starting index of each block.
        for i in range(0, n, k):
            # Extract the k-length substring (block) starting at index i.
            block = word[i : i + k]
            # Increment the count for this specific block string in the Counter.
            block_counts[block] += 1
            
        # Find the frequency of the most frequent block.
        # The goal is to make all blocks identical to this most frequent block
        # to minimize the number of operations.
        max_freq = 0
        # The constraints (n >= 1, k >= 1, k divides n) ensure that m = n/k >= 1.
        # This means there will always be at least one block, and block_counts will not be empty.
        # Therefore, calling max() on block_counts.values() is safe.
        if block_counts: # This check is technically redundant due to constraints but ensures robustness.
            max_freq = max(block_counts.values())
            
        # The total number of blocks is m.
        # The number of blocks that already match the target (most frequent) block is max_freq.
        # The number of blocks that need to be changed is the total number of blocks minus
        # the count of the most frequent block.
        # Each operation can change one block to match the target block.
        min_operations = m - max_freq
        
        return min_operations