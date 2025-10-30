class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        n = len(s)
        m = len(pattern)
        
        for i in range(n - m + 1):
            mismatch_count = 0
            for j in range(m):
                if s[i + j] != pattern[j]:
                    mismatch_count += 1
                if mismatch_count > 1:
                    break
            if mismatch_count <= 1:
                return i
        
        return -1