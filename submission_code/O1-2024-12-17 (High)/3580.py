class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        n, m = len(s), len(pattern)
        # If the pattern is longer than s, no valid substring can exist.
        if m > n:
            return -1
        
        # Check every possible starting index i in s where a substring
        # of length m can fit.
        for i in range(n - m + 1):
            # Count the number of mismatches between s[i..i+m-1] and pattern[0..m-1].
            mismatch_count = 0
            for j in range(m):
                if s[i + j] != pattern[j]:
                    mismatch_count += 1
                    # If more than one mismatch, no need to check further.
                    if mismatch_count > 1:
                        break
            # If at most one mismatch was found, return this starting index.
            if mismatch_count <= 1:
                return i
        
        # If no valid starting index was found, return -1.
        return -1