class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        N = len(s)
        M = len(pattern)
        if M >= N:
            return -1  # Pattern cannot be a substring of s
        
        # Calculate mismatch count for the first window
        mismatch = 0
        for j in range(M):
            if s[j] != pattern[j]:
                mismatch += 1
        if mismatch <= 1:
            return 0
        
        # Slide the window from left to right
        for i in range(1, N - M + 1):
            # Remove the mismatch count of the leftmost character
            if s[i - 1] != pattern[0]:
                mismatch -= 1
            # Add the mismatch count of the new rightmost character
            if s[i + M - 1] != pattern[M - 1]:
                mismatch += 1
            # Check if current window has at most one mismatch
            if mismatch <= 1:
                return i
        
        return -1