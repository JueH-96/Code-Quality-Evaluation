class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        if n % k != 0:
            return -1  # This should not happen as per the constraints
        
        # Group the string into chunks of size k
        chunks = [word[i:i + k] for i in range(0, n, k)]
        
        # Count the frequency of each chunk
        chunk_count = {}
        for chunk in chunks:
            if chunk in chunk_count:
                chunk_count[chunk] += 1
            else:
                chunk_count[chunk] = 1
        
        # Find the most frequent chunk
        max_count = max(chunk_count.values())
        
        # The minimum number of operations is the total number of chunks minus the most frequent chunk
        return len(chunks) - max_count