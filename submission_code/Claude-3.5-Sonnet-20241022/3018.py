class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        def next_char(c):
            return chr((ord(c) - ord('a') + 1) % 26 + ord('a'))
            
        i = 0  # pointer for str2
        j = 0  # pointer for str1
        
        while i < len(str2) and j < len(str1):
            if str1[j] == str2[i] or next_char(str1[j]) == str2[i]:
                i += 1
            j += 1
            
        return i == len(str2)