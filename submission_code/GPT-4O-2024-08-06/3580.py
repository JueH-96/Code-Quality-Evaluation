class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        m, n = len(s), len(pattern)
        
        # Iterate over each possible starting index in s
        for i in range(m - n + 1):
            # Count the number of mismatches between the substring of s and pattern
            mismatch_count = 0
            for j in range(n):
                if s[i + j] != pattern[j]:
                    mismatch_count += 1
                # If more than one mismatch, break early
                if mismatch_count > 1:
                    break
            
            # If we found a substring with at most one mismatch, return the starting index
            if mismatch_count <= 1:
                return i
        
        # If no such substring is found, return -1
        return -1