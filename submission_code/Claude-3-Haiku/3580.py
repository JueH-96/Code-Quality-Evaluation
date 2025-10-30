class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        n, m = len(s), len(pattern)
        if m > n:
            return -1
        
        def is_almost_equal(i):
            diff = 0
            for j in range(m):
                if s[i+j] != pattern[j]:
                    diff += 1
                    if diff > 1:
                        return False
            return True
        
        for i in range(n-m+1):
            if is_almost_equal(i):
                return i
        
        return -1