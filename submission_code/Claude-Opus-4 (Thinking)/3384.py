class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        from collections import Counter
        
        # Divide the string into chunks of size k
        chunks = [word[i:i+k] for i in range(0, len(word), k)]
        
        # Count frequency of each chunk
        freq = Counter(chunks)
        
        # Minimum operations = total chunks - max frequency
        # We keep the most frequent chunk and replace all others
        return len(chunks) - max(freq.values())