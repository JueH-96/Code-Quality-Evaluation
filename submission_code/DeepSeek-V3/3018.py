class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        n = len(str1)
        m = len(str2)
        if m > n:
            return False
        i = 0
        j = 0
        while i < n and j < m:
            current_char = str1[i]
            next_char = chr(ord(current_char) + 1) if current_char != 'z' else 'a'
            if str2[j] == current_char or str2[j] == next_char:
                j += 1
            i += 1
        return j == m