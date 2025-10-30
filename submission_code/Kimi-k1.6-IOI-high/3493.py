class Solution:
    def maxOperations(self, s: str) -> int:
        res = 0
        ones = 0
        current_zeros = 0
        for c in s:
            if c == '0':
                current_zeros += 1
            else:
                res += current_zeros * ones
                current_zeros = 0
                ones += 1
        return res