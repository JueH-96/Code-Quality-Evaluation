class Solution:
    def maxOperations(self, s: str) -> int:
        n = len(s)
        operations = 0
        i = 0
        while i < n:
            if s[i] == '1':
                j = i + 1
                while j < n and s[j] == '0':
                    j += 1
                if j < n and s[j] == '1':
                    operations += j - i - 1
                    s = s[:i] + '0' + s[i+1:j] + '1' + s[j+1:]
                else:
                    operations += j - i - 1
                    s = s[:i] + '0' + s[i+1:j] + '1' + s[j:]
            i += 1
        return operations