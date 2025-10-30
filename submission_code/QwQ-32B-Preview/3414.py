class Solution:
    def waysToReachStair(self, k: int) -> int:
        return 1 << (k + 1)