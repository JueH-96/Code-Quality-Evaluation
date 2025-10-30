from collections import Counter

class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        # Split the word into chunks of size k
        chunks = [word[i:i+k] for i in range(0, len(word), k)]
        # Count the frequency of each chunk
        counts = Counter(chunks)
        # The maximum frequency of any chunk
        max_freq = max(counts.values(), default=0)
        # The minimum operations needed is total chunks minus the most frequent one
        return len(chunks) - max_freq