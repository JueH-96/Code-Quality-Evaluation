class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        if n == 0:
            return -1
        
        contrib = [2 * x - 1 for x in possible]
        prefix = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix[i] = prefix[i - 1] + contrib[i - 1]
        
        suffix = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix[i] = suffix[i + 1] + contrib[i]
        
        for k in range(1, n):
            if prefix[k] > suffix[k]:
                return k
        
        return -1