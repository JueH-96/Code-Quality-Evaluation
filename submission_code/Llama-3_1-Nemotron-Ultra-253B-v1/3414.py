class Solution:
    def waysToReachStair(self, k: int) -> int:
        if k == 0:
            return 2
        m = k + 1
        return 2 ** (m.bit_length())