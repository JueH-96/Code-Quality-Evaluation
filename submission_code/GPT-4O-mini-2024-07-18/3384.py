class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        operations = 0
        
        # We will check each position in the k-length segments
        for i in range(k):
            # Count frequency of each character in the current position across all segments
            freq = {}
            for j in range(i, n, k):
                if word[j] in freq:
                    freq[word[j]] += 1
                else:
                    freq[word[j]] = 1
            
            # Find the maximum frequency of any character in this position
            max_freq = max(freq.values(), default=0)
            # The number of operations needed for this position is the number of characters
            # minus the maximum frequency character count
            operations += (n // k) - max_freq
        
        return operations