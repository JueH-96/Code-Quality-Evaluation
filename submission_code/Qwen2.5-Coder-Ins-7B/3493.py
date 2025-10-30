class Solution:
    def maxOperations(self, s: str) -> int:
        count = 0
        ones = 0
        for char in s:
            if char == '1':
                ones += 1
            elif ones > 0:
                count += ones
        return count