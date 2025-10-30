class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        i = 0
        j = 0
        while i < len(str1) and j < len(str2):
            if str1[i] == str2[j] or str1[i] == (chr(ord(str2[j]) - 1) if str2[j] != 'a' else 'z'):
                j += 1
            i += 1
        return j == len(str2)