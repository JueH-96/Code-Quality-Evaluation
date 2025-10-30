class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        len_s = len(s)
        len_p = len(pattern)
        
        if len_p > len_s:
            return -1
        
        mismatches = 0
        # Initialize mismatch count for the first window
        for j in range(len_p):
            if s[j] != pattern[j]:
                mismatches += 1
        
        if mismatches <= 1:
            return 0
        
        # Slide the window
        for i in range(1, len_s - len_p + 1):
            # Remove the leftmost character of the previous window
            if s[i - 1] != pattern[0]:
                mismatches -= 1
            # Add the new rightmost character of the current window
            if s[i + len_p - 1] != pattern[-1]:
                mismatches += 1
            # Check if current window has at most one mismatch
            if mismatches <= 1:
                return i
        return -1