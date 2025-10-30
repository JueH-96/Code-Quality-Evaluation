class Solution:
    def maxOperations(self, s: str) -> int:
        ones = 0
        res = 0
        for c in reversed(s):
            if c == '1':
                ones += 1
            else:
                if ones >= 1:
                    res += 1
                    ones -= 1
        return res