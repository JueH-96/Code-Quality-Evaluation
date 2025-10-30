class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        n = len(str1)
        m = len(str2)
        i = 0
        j = 0
        while i < n and j < m:
            if str1[i] == str2[j] or ord(str1[i]) + 1 == ord(str2[j]) or (str1[i] == 'z' and str2[j] == 'a'):
                j += 1
            i += 1
        return j == m