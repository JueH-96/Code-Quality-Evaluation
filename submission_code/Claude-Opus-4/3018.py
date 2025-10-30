class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        i = 0  # pointer for str1
        j = 0  # pointer for str2
        
        while i < len(str1) and j < len(str2):
            # Check if current character in str1 matches current character in str2
            # either directly or after incrementing
            if str1[i] == str2[j] or chr((ord(str1[i]) - ord('a') + 1) % 26 + ord('a')) == str2[j]:
                j += 1
            i += 1
        
        # If we've matched all characters in str2, it's a subsequence
        return j == len(str2)