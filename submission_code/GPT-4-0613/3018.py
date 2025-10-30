class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        if len(str2) > len(str1):
            return False
        str1 = list(str1)
        str2 = list(str2)
        j = 0
        for i in range(len(str1)):
            if j < len(str2) and str1[i] >= str2[j]:
                str1[i] = str2[j]
                j += 1
        return j == len(str2)