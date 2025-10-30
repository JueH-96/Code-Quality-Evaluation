class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        n = len(s)
        m = len(pattern)
        
        if m > n:
            return -1
        
        min_index = -1
        
        for i in range(n - m + 1):
            substring = s[i:i+m]
            diff_count = 0
            
            for j in range(m):
                if substring[j] != pattern[j]:
                    diff_count += 1
            
            if diff_count <= 1:
                if min_index == -1:
                    min_index = i
                else:
                    min_index = min(min_index, i)
        
        return min_index