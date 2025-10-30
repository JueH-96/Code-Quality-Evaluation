class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        j = 0
        m = len(str2)
        for c in str1:
            if j == m:
                break
            if c == str2[j]:
                j += 1
            else:
                next_char = chr((ord(c) - ord('a') + 1) % 26 + ord('a'))
                if next_char == str2[j]:
                    j += 1
        return j == m