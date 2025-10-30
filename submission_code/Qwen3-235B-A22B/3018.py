class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        j = 0
        m = len(str2)
        n = len(str1)
        if m > n:
            return False
        for c in str1:
            if j >= m:
                break
            current = str2[j]
            if c == current:
                j += 1
            else:
                # compute modified character
                mod_c = (ord(c) - ord('a') + 1) % 26
                mod_char = chr(mod_c + ord('a'))
                if mod_char == current:
                    j += 1
        return j == m