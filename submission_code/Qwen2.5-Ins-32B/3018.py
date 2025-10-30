class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        def next_char(c):
            return 'a' if c == 'z' else chr(ord(c) + 1)
        
        j = 0
        for i in range(len(str1)):
            if j < len(str2) and (str1[i] == str2[j] or next_char(str1[i]) == str2[j]):
                j += 1
        return j == len(str2)