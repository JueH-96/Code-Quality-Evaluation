class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        num_periods = n // k
        
        first_period = word[:k]
        
        operations = 0
        for i in range(k, n, k):
            if word[i:i+k] != first_period:
                operations += 1
        
        return operations