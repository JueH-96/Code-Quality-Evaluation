class Solution:
    def waysToReachStair(self, k: int) -> int:
        return pow(2, bin(k).count('1') + 1)