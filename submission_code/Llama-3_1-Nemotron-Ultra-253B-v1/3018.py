class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        j = 0
        n = len(str2)
        if n == 0:
            return True
        for c in str1:
            if j >= n:
                break
            current = c
            next_char = chr((ord(c) - ord('a') + 1) % 26 + ord('a'))
            if current == str2[j] or next_char == str2[j]:
                j += 1
        return j == n