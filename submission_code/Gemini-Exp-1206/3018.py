class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        n = len(str1)
        m = len(str2)
        
        def is_subsequence(s1, s2):
            i = 0
            j = 0
            while i < len(s1) and j < len(s2):
                if s1[i] == s2[j]:
                    j += 1
                i += 1
            return j == len(s2)

        if is_subsequence(str1, str2):
            return True

        for i in range(1 << n):
            temp = list(str1)
            for j in range(n):
                if (i >> j) & 1:
                    temp[j] = chr(((ord(temp[j]) - ord('a') + 1) % 26) + ord('a'))
            
            if is_subsequence("".join(temp), str2):
                return True

        return False