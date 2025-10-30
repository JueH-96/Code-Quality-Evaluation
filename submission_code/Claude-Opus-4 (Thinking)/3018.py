class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        i = 0  # pointer for str1
        j = 0  # pointer for str2
        
        while i < len(str1) and j < len(str2):
            # Check if str1[i] matches str2[j] directly or after increment
            if str1[i] == str2[j] or chr((ord(str1[i]) - ord('a') + 1) % 26 + ord('a')) == str2[j]:
                j += 1  # Move to next character in str2
            i += 1  # Always move to next character in str1
        
        return j == len(str2)  # All characters in str2 should be matched