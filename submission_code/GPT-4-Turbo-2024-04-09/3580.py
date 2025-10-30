class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        len_s = len(s)
        len_p = len(pattern)
        
        # Function to check if pattern can be converted to substring of s starting at index i with at most one change
        def is_almost_equal(i):
            mismatch_count = 0
            for j in range(len_p):
                if s[i + j] != pattern[j]:
                    mismatch_count += 1
                    if mismatch_count > 1:
                        return False
            return True
        
        # Check each possible starting index in s where the full pattern can fit
        for i in range(len_s - len_p + 1):
            if is_almost_equal(i):
                return i
        
        return -1  # If no valid starting index is found