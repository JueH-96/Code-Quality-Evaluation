class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        i, j, m, n = 0, 0, len(str1), len(str2)
        while i < m and j < n:
            if str1[i] == str2[j]:
                j += 1
            elif ord(str1[i]) - ord('a') + 1 != ord(str2[j]) - ord('a'):
                return False
            i += 1
        return j == n