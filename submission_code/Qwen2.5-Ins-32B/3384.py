class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        base = word[:k]
        count = 0
        
        for i in range(k, n, k):
            if word[i:i+k] != base:
                count += 1
        
        return count