class Solution:
    def maxOperations(self, s: str) -> int:
        ones = 0
        operations = 0
        for char in s:
            if char == '1':
                ones += 1
            elif ones > 0:
                operations += ones
        return operations