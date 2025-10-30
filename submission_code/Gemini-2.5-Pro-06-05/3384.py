import collections

class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        
        # Count the frequency of each k-length chunk using a Counter.
        # A generator expression iterates over the chunks efficiently without
        # storing them all in a list first.
        chunk_counts = collections.Counter(word[i:i + k] for i in range(0, n, k))
        
        # The total number of chunks is n/k, since k is guaranteed to divide n.
        total_chunks = n // k
        
        # To make the word k-periodic, all chunks must be identical.
        # The operation allows us to replace any chunk with any other existing chunk.
        # To minimize operations, we should make all chunks equal to the most frequent one.
        
        # The number of chunks that we don't need to change is the frequency
        # of the most common chunk. The problem constraints (n >= 1) ensure that
        # chunk_counts will not be empty, so max() is safe to call.
        max_frequency = max(chunk_counts.values())
        
        # The minimum number of operations is the total number of chunks minus
        # the number of chunks that already match our target (the most frequent ones).
        return total_chunks - max_frequency