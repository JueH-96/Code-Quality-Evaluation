class Solution:
    def waysToReachStair(self, k: int) -> int:
        return 2 ** (k + 1)