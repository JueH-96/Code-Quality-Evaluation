class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        mod = 10**9 + 7
        res = 0
        for i in range(k):
            res += nums[i]
            res %= mod
        return res