class Solution:
    def minChanges(self, s: str) -> int:
        l, r = 0, len(s) - 1
        res = 0
        
        while l < r:
            if s[l] != s[r]:
                res += 1
            l += 1
            r -= 1
            
        return res