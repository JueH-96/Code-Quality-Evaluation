class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        if n == 0 or k == 0:
            return 0
        
        num_periods = n // k
        
        first_period = word[:k]
        
        count = 0
        for i in range(k, n, k):
            if word[i:i+k] != first_period:
                count += 1
        
        return count