class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        n = len(s)
        m = len(pattern)
        
        # Check each possible starting position
        for i in range(n - m + 1):
            differences = 0
            
            # Compare substring s[i:i+m] with pattern
            for j in range(m):
                if s[i + j] != pattern[j]:
                    differences += 1
                    # Early termination if more than 1 difference
                    if differences > 1:
                        break
            
            # If at most 1 difference, we found our answer
            if differences <= 1:
                return i
        
        return -1