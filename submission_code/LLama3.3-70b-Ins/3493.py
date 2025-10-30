class Solution:
    def maxOperations(self, s: str) -> int:
        operations = 0
        i = 0
        while i < len(s) - 1:
            if s[i] == '1' and s[i + 1] == '0':
                operations += 1
                j = i + 1
                while j < len(s) - 1 and s[j + 1] == '0':
                    j += 1
                s = s[:i] + '0' + s[i + 1:j + 1] + '1' + s[j + 2:]
                i = 0
            else:
                i += 1
        return operations