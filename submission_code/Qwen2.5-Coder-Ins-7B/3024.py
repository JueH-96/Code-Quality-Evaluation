class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        MOD = 10**9 + 7
        n = len(s)
        
        if s == t:
            if k % 2 == 0:
                return 1
            else:
                return 0
        
        if k == 1:
            return 0
        
        count = 0
        for i in range(n):
            if s[i:] + s[:i] == t:
                count += 1
        
        if count == 0:
            return 0
        
        return pow(count, k, MOD)