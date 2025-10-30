class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        num_segments = n // k
        min_operations = 0
        
        for i in range(k):
            count = {}
            for j in range(i, n, k):
                count[word[j]] = count.get(word[j], 0) + 1
            
            max_freq = max(count.values())
            min_operations += (num_segments - max_freq)
        
        return min_operations