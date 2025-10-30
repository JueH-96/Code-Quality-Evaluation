class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        if n == k:
            return 0
        
        # Check if the word is already k-periodic
        s = word[:k]
        if all(word[i:i+k] == s for i in range(0, n, k)):
            return 0
        
        # Find the minimum number of operations required
        min_ops = float('inf')
        for i in range(0, n, k):
            for j in range(0, n, k):
                if i != j:
                    new_word = word[:i] + word[j:j+k] + word[i+k:]
                    ops = 1
                    if all(new_word[x:x+k] == new_word[x+k:x+2*k] for x in range(0, len(new_word)-k, k)):
                        min_ops = min(min_ops, ops)
        
        return min_ops if min_ops != float('inf') else -1