class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        def next_char(c):
            return chr((ord(c) - ord('a') + 1) % 26 + ord('a'))
        
        i, j = 0, 0
        while i < len(str1) and j < len(str2):
            if str1[i] == str2[j] or next_char(str1[i]) == str2[j]:
                j += 1
            i += 1
        
        return j == len(str2)