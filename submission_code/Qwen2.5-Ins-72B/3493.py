class Solution:
    def maxOperations(self, s: str) -> int:
        operations = 0
        ones_count = 0
        for i in range(len(s) - 1, -1, -1):
            if s[i] == '1':
                ones_count += 1
            elif ones_count > 0:
                operations += ones_count
        return operations