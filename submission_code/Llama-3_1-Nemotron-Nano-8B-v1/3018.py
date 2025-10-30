class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        i = j = 0
        n, m = len(str1), len(str2)
        if m == 0:
            return True
        while i < n and j < m:
            current_char = str2[j]
            if str1[i] == current_char:
                j += 1
                i += 1
            else:
                # Check if incrementing str1[i] would match current_char
                if (ord(str1[i]) - ord('a') + 1) % 26 == (ord(current_char) - ord('a')):
                    j += 1
                    i += 1
                else:
                    i += 1
        return j == m