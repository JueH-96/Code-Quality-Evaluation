class Solution:
    def maxOperations(self, s: str) -> int:
        count = 0
        i = 0
        while i < len(s) - 1:
            if s[i] == '1' and s[i+1] == '0':
                j = i + 1
                while j < len(s) and s[j] == '0':
                    j += 1
                count += 1
                s = s[:i] + s[i+1:j] + s[i] + s[j:]
                i = j
            else:
                i += 1
        return count