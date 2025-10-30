class Solution:
    def maxOperations(self, s: str) -> int:
        ones = 0
        result = 0
        for char in s:
            if char == '1':
                ones += 1
            else:
                result += ones
        return result