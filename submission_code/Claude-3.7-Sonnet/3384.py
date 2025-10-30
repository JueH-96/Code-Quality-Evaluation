class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        from collections import Counter
        
        # Divide the string into chunks of size k
        chunks = [word[i:i+k] for i in range(0, len(word), k)]
        
        # Count the frequency of each unique chunk
        chunk_counts = Counter(chunks)
        
        # Find the count of the most frequent chunk
        most_frequent_count = chunk_counts.most_common(1)[0][1]
        
        # The number of operations needed is the total number of chunks
        # minus the frequency of the most common chunk
        return len(chunks) - most_frequent_count