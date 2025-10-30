class Solution:
    def waysToReachStair(self, k: int) -> int:
        if k == 0:
            return 2
        if k == 1:
            return 4
        ways = [0] * (k + 2)
        ways[1] = 1
        ways[2] = 2
        for i in range(3, k + 2):
            ways[i] = ways[i - 1] + ways[i - 2]
        return ways[k + 1]