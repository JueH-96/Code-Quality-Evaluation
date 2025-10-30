class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        # Initialize pointers for str1 and str2
        i, j = 0, 0
        
        # Iterate over both strings
        while i < len(str1) and j < len(str2):
            # Check if the current character in str1 can be incremented to match the current character in str2
            if (ord(str1[i]) - ord('a') + 1) % 26 == ord(str2[j]) - ord('a'):
                j += 1  # Move to the next character in str2
            i += 1  # Move to the next character in str1
        
        # If we've matched all characters in str2, then it's a subsequence
        return j == len(str2)