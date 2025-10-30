class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        n = len(s)
        m = len(pattern)
        
        for i in range(n - m + 1):
            differences = 0
            for j in range(m):
                if s[i + j] != pattern[j]:
                    differences += 1
                    if differences > 1:
                        break  # Early termination optimization
            
            if differences <= 1:
                return i
        
        return -1