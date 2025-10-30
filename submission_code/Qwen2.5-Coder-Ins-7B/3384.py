class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        groups = [word[i:i+k] for i in range(0, n, k)]
        target = groups[0]
        operations = 0
        
        for group in groups[1:]:
            if group != target:
                operations += 1
        
        return operations