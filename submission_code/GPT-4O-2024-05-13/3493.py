class Solution:
    def maxOperations(self, s: str) -> int:
        operations = 0
        i = 0
        while i < len(s) - 1:
            if s[i] == '1' and s[i + 1] == '0':
                operations += 1
                i += 2  # Skip the next character as it has been moved
            else:
                i += 1
        return operations