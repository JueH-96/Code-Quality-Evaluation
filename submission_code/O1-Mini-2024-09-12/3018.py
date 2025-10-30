class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        i = 0
        j = 0
        n = len(str1)
        m = len(str2)
        while i < n and j < m:
            if str1[i] == str2[j]:
                j += 1
            elif chr(((ord(str1[i]) - ord('a') + 1) % 26) + ord('a')) == str2[j]:
                j += 1
            i += 1
        return j == m