class Solution:
    def maxOperations(self, s: str) -> int:
        ones = zeros = operations = 0
        for char in s:
            if char == '1':
                ones += 1
            elif ones > 0:
                ones -= 1
                zeros += 1
            if zeros > 0:
                zeros -= 1
                operations += 1
        return operations