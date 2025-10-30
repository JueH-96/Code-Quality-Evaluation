class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        def isSubsequence(s1, s2):
            i = 0
            j = 0
            while i < len(s1) and j < len(s2):
                if s1[i] == s2[j]:
                    j += 1
                i += 1
            return j == len(s2)

        if isSubsequence(str1, str2):
            return True

        for i in range(1 << len(str1)):
            temp_str1 = list(str1)
            for j in range(len(str1)):
                if (i >> j) & 1:
                    temp_str1[j] = chr(((ord(temp_str1[j]) - ord('a') + 1) % 26) + ord('a'))
            
            if isSubsequence("".join(temp_str1), str2):
                return True
        return False