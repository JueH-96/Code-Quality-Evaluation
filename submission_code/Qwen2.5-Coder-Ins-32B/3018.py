class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        i, j = 0, 0
        len1, len2 = len(str1), len(str2)
        
        while i < len1 and j < len2:
            if str1[i] == str2[j] or (ord(str1[i]) - ord('a') + 1) % 26 == ord(str2[j]) - ord('a'):
                j += 1
            i += 1
        
        return j == len2