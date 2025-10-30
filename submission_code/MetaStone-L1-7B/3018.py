class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        i = j = 0
        n, m = len(str1), len(str2)
        while i < n and j < m:
            c = str2[j]
            # Look for c or c+1 in str1 starting from i
            found = False
            while i < n:
                current_char = str1[i]
                if current_char == c:
                    found = True
                    break
                elif current_char == chr(ord(c) + 1) and c != 'z':
                    found = True
                    break
                i += 1
            if found:
                i += 1
                j += 1
            else:
                return False
        return j == m