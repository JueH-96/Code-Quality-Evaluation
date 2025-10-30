class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        n1 = len(str1)
        n2 = len(str2)
        i = 0
        j = 0
        while i < n1 and j < n2:
            if str1[i] == str2[j]:
                j += 1
            elif chr(((ord(str1[i]) - ord('a') + 1) % 26) + ord('a')) == str2[j]:
                j += 1
            i += 1
        return j == n2