class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        prefix, suffix = [0] * n, [0] * n
        for i in range(1, n):
            if s[i] != s[i - 1]:
                prefix[i] = 1
            else:
                prefix[i] = 0
            if i > 0:
                prefix[i] += prefix[i - 1]
        
        for i in range(n - 2, -1, -1):
            if s[i] != s[i + 1]:
                suffix[i] = 1
            else:
                suffix[i] = 0
            if i + 1 < n:
                suffix[i] += suffix[i + 1]
        
        result = min(prefix[-1], suffix[0])
        for k in range(1, n - 1):
            result = min(result, prefix[k - 1] + suffix[k])
            
        return result