class Solution:
    def maxOperations(self, s: str) -> int:
        count_ones = 0
        operations = 0
        for c in s:
            if c == '1':
                count_ones += 1
            elif c == '0':
                operations += count_ones
        return operations