class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        s = set()
        i, j = 0, 0
        while i < len(str1) and j < len(str2):
            if str1[i] == str2[j]:
                # Match without increment
                j += 1
            elif (ord(str1[i]) + 1 - ord('a')) % 26 == ord(str2[j]) - ord('a'):
                # Match with increment
                s.add(i)
                j += 1
            i += 1
        return j == len(str2)