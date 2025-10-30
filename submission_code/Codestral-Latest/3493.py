class Solution:
    def maxOperations(self, s: str) -> int:
        n = len(s)
        operations = 0
        i = 0

        while i < n - 1:
            if s[i] == '1' and s[i + 1] == '0':
                operations += 1
                # Move the '1' to the right until it reaches the end or another '1'
                j = i
                while j < n - 1 and s[j + 1] == '0':
                    j += 1
                # Swap s[i] with s[j]
                s = s[:i] + s[i + 1:j + 1] + '1' + s[j + 1:]
                i = j + 1
            else:
                i += 1

        return operations