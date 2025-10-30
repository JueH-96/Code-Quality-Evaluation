class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        pos = [i for i, num in enumerate(nums) if num == 1]
        m = len(pos)
        if m == 0:
            return 0
        if m == 1:
            return 1
        ways = 1
        for i in range(1, m):
            gap = pos[i] - pos[i - 1] - 1
            ways = ways * (gap + 1) % MOD
        return ways