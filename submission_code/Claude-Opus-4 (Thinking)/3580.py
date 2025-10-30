class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        n = len(s)
        m = len(pattern)
        
        # Iterate through all possible starting positions
        for i in range(n - m + 1):
            # Count differences between substring and pattern
            differences = 0
            for j in range(m):
                if s[i + j] != pattern[j]:
                    differences += 1
                    if differences > 1:
                        break
            
            # If at most 1 difference, we found our answer
            if differences <= 1:
                return i
        
        return -1