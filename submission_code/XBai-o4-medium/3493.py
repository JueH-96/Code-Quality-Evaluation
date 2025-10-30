class Solution:
    def maxOperations(self, s: str) -> int:
        count_1 = 0
        res = 0
        for c in s:
            if c == '1':
                count_1 += 1
            else:
                res += count_1
        return res