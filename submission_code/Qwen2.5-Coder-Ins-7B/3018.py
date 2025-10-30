class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        j = 0
        for i in range(len(str1)):
            if (ord(str1[i]) - ord('a') + 1) % 26 == (ord(str2[j]) - ord('a')) % 26:
                j += 1
                if j == len(str2):
                    return True
        return False