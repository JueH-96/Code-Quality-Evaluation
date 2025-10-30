class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        n, m = len(str1), len(str2)
        if m > n:
            return False
        i = j = 0
        while i < n and j < m:
            current_char = str1[i]
            next_char = chr(ord('a') + (ord(current_char) - ord('a') + 1) % 26)
            if current_char == str2[j] or next_char == str2[j]:
                j += 1
            i += 1
        return j == m